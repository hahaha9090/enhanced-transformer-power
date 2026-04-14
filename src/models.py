# 导入深度学习库 torch
import torch

# 导入数学库
import math

# 设置计算设备：如果有 CUDA 则使用 GPU，否则使用 CPU
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


# 定义位置编码类，继承自 torch.nn.Module
class PositionalEncoding(torch.nn.Module):
    # 初始化时不记录梯度
    @torch.no_grad()
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        PositionalEncoding 类的构造函数

        参数:
            d_model: 模型输入的维度
            dropout: 随机失活率，默认为 0.1
            max_len: 最大序列长度，默认为 5000
        """

        # 调用父类构造函数
        super().__init__()

        # 定义 Dropout 层，用于防止过拟合
        self.dropout = torch.nn.Dropout(p=dropout)

        # 生成位置序列 [0, 1, 2, ..., max_len-1]，形状为 (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)
        # 计算分母项，用于缩放不同频率的正弦和余弦函数
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # 初始化位置编码张量，形状为 (max_len, 1, d_model)，并注册为参数
        self.pe: torch.Tensor = torch.nn.Parameter(torch.zeros(max_len, 1, d_model))
        # 在偶数维度上填充正弦值
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        # 在奇数维度上填充余弦值
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    # 前向传播函数
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法

        参数:
            inputs: 输入张量

        返回:
            输出张量
        """

        # 将位置编码加到输入张量上，截取与输入长度相同的部分
        outputs = inputs + self.pe[: inputs.size(0)]
        # 应用 Dropout 并返回结果
        return self.dropout(outputs)


# 定义基于 Transformer 的电力预测基础模型
class BaseDailyElectricTransformer(torch.nn.Module):
    """
    该类基于 Transformer Encoder 和多层感知机 (MLP)。每一天（24个价格）被视为一个“单词”。

    参数:
        values_embeddings: 过去值的嵌入层
        positional_encoding: Transformer Encoder 的位置编码
        transformer_encoder: Transformer 编码器
        features_embeddings: 特征的嵌入层
        mlp: 模型的多层感知机
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 8,
        dim_feedforward: int = 128,
        num_layers: int = 6,
        normalize_first: bool = False,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        """
        BaseDailyElectricTransformer 类的构造函数

        参数:
            embedding_dim: 嵌入维度，默认为 32
            num_heads: Transformer 的多头注意力头数，默认为 8
            dim_feedforward: 前馈网络的维度，默认为 128
            num_layers: Transformer 的层数，默认为 6
            normalize_first: 是否在 Transformer Encoder 中先进行归一化，默认为 False
            dropout: 随机失活率，默认为 0.2
            activation: 使用的激活函数，默认为 "relu"
        """

        # 调用父类构造函数
        super().__init__()

        # 根据配置选择激活函数
        activation_function: torch.nn.Module
        if activation == "relu":
            activation_function = torch.nn.ReLU()
        else:
            activation_function = torch.nn.GELU()

        # 定义输入数值的嵌入层：将 24 小时的数据映射到 embedding_dim 维度
        self.values_embeddings = torch.nn.Sequential(
            torch.nn.Linear(24, embedding_dim),
            activation_function,
        )

        # 初始化位置编码层
        self.positional_encoding = PositionalEncoding(embedding_dim)
        # 定义 Transformer 编码器层
        encoder_layers = torch.nn.TransformerEncoderLayer(
            embedding_dim,
            num_heads,
            dim_feedforward,
            dropout,
            batch_first=True, # 输入形状为 [batch, seq, feature]
            norm_first=normalize_first,
            activation=activation,
        )
        # 构建多层 Transformer 编码器
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layers, num_layers
        )

        # 定义当前特征（如时间、天气等）的嵌入层：将 48 维特征映射到 embedding_dim
        self.features_embeddings = torch.nn.Sequential(
            torch.nn.Linear(48, embedding_dim), activation_function
        )

        # 定义多层感知机 (MLP) 用于最终输出
        self.mlp = torch.nn.Sequential(
            # 对拼接后的向量进行层归一化
            torch.nn.LayerNorm(2 * embedding_dim),
            # 全连接层：从 2*embedding_dim 映射到 dim_feedforward
            torch.nn.Linear(2 * embedding_dim, dim_feedforward),
            # Dropout 层
            torch.nn.Dropout(dropout),
            # 激活函数
            activation_function,
            # 输出层：映射回 24 小时的数据
            torch.nn.Linear(dim_feedforward, 24),
        )

    # 前向传播逻辑
    def forward(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        计算 BaseDailyElectricTransformer 对象的输出。

        参数:
            values: 过去的电价数据。维度: [batch, sequence length, 1]
            features: 当前的特征数据。维度: [batch, 24, 2]

        返回:
            预测的电价批次。维度: [batch, sequence_length]
        """

        # 将过去的电价数据重塑并计算嵌入表示
        # 输入维度变换: [batch, seq_len, 1] -> [batch, days, 24]
        values_embeddings = self.values_embeddings(
            values.reshape((values.size(0), -1, 24))
        )

        # 应用位置编码
        transformer_inputs = self.positional_encoding(values_embeddings)
        # 通过 Transformer 编码器处理序列
        transformer_outputs = self.transformer_encoder(transformer_inputs)

        # 将当前特征数据重塑并计算嵌入表示
        # 输入维度变换: [batch, 24, 2] -> [batch, 1, 48]
        features_embeddings = self.features_embeddings(
            features.reshape(features.size(0), -1, 48)
        )

        # 将特征嵌入与 Transformer 的输出在特征维度上进行拼接
        mlp_inputs = torch.concat((features_embeddings, transformer_outputs), dim=2)
        # 通过多层感知机计算最终预测值
        outputs = self.mlp(mlp_inputs)

        # 重塑输出形状并返回
        return outputs.reshape((values.size(0), -1))


# 定义对称平均绝对百分比误差 (sMAPE) 损失函数
class sMAPELoss(torch.nn.Module):
    def __init__(self) -> None:
        """
        sMAPELoss 的构造函数
        """

        # 调用父类构造函数
        super().__init__()

    # 前向传播计算损失
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法

        参数:
            predictions: 模型预测值张量
            targets: 真实目标值张量

        返回:
            计算出的损失值张量
        """

        # 计算分子部分：2 * |真实值 - 预测值|
        differences = 2 * torch.abs(targets - predictions)
        # 计算分母部分：|真实值| + |预测值|，并求平均损失
        loss_value = torch.mean(
            differences / (torch.abs(targets) + torch.abs(predictions))
        )

        # 返回最终的损失值
        return loss_value
