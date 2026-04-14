# 导入数据处理库 pandas
import pandas as pd
# 导入深度学习库 torch 及其相关组件
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 导入数值计算库 numpy
import numpy as np
# 从 epftoolbox 库中导入各种评估指标函数
from epftoolbox.evaluation import MAE, RMSE, MAPE, sMAPE, DM

# 导入类型提示相关的库
from typing import Optional, List


# 启用梯度计算装饰器，用于训练函数
@torch.enable_grad()
def train(
    train_data: DataLoader,
    val_data: DataLoader,
    model: torch.nn.Module,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    clip_gradients: Optional[float],
    epochs: int,
    writer: SummaryWriter,
    save_path: str,
) -> None:
    """
    该函数负责训练模型。

    参数:
        train_data: 训练数据的 DataLoader
        val_data: 验证数据的 DataLoader
        model: 待训练的模型
        mean: 目标变量的均值，用于反标准化
        std: 目标变量的标准差，用于反标准化
        loss: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        clip_gradients: 梯度裁剪阈值
        epochs: 训练的总轮数
        writer: TensorBoard 记录器
        save_path: 模型保存路径
    """

    # 遍历每一个训练周期
    for epoch in range(epochs):
        # 打印当前周期数
        print(f"Epoch: {epoch}")

        # 将模型设置为训练模式
        model.train()

        # 初始化用于存储各项指标和损失的列表
        mae_vector = []
        rmse_vector = []
        mape_vector = []
        smape_vector = []
        losses = []

        # 遍历训练数据批次
        for inputs in train_data:
            # 准备数据：转换为浮点数
            inputs = inputs.float()
            # 提取过去的价格数据并增加维度，维度: [batch, seq_len, 1]
            values = inputs[:, :-24, 0].unsqueeze(2)
            # 提取当前的外部特征数据，维度: [batch, 24, 2]
            features = inputs[:, 24:, 1:3]
            # 提取目标电价数据
            targets = inputs[:, 24:, 0]

            # 计算模型输出和损失值
            outputs = model(values, features)
            loss_value = loss(outputs, targets)

            # 反向传播与参数优化
            optimizer.zero_grad() # 清除历史梯度
            loss_value.backward() # 计算当前梯度
            # 如果设置了梯度裁剪，则应用裁剪
            if clip_gradients is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
            optimizer.step() # 更新模型参数

            # 记录当前批次的损失值（转换为 Python 标量）
            losses.append(loss_value.item())
            
            # 计算并记录各项评估指标（需先进行反标准化）
            # 反标准化公式：scaled_value * std + mean
            target_numpy = targets.detach().cpu().numpy() * std + mean
            output_numpy = outputs.detach().cpu().numpy() * std + mean
            
            mae_vector.append(MAE(target_numpy, output_numpy))
            rmse_vector.append(RMSE(target_numpy, output_numpy))
            mape_vector.append(MAPE(target_numpy, output_numpy))
            smape_vector.append(sMAPE(target_numpy, output_numpy))

        # 将训练阶段的平均指标记录到 TensorBoard
        writer.add_scalar("loss", np.mean(losses), epoch)
        writer.add_scalar("MAE/train", np.mean(mae_vector), epoch)
        writer.add_scalar("RMSE/train", np.mean(rmse_vector), epoch)
        writer.add_scalar("MAPE/train", np.mean(mape_vector), epoch)
        writer.add_scalar("sMAPE/train", np.mean(smape_vector), epoch)
        # 记录当前学习率
        if optimizer.param_groups[0]["lr"] is not None:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # 如果存在调度器，则更新学习率
        if scheduler is not None:
            scheduler.step()

        # 验证阶段：将模型设置为评估模式
        model.eval()
        # 禁用梯度计算以节省内存和计算资源
        with torch.no_grad():
            # 初始化验证阶段的指标列表
            mae_vector = []
            rmse_vector = []
            mape_vector = []
            smape_vector = []

            # 遍历验证数据批次
            for inputs in val_data:
                # 准备验证数据
                inputs = inputs.float()
                values = inputs[:, :-24, 0].unsqueeze(2)
                features = inputs[:, 24:, 1:3]
                targets = inputs[:, -24:, 0]

                # 计算模型输出并截取最后 24 小时的预测结果
                outputs = model(values, features)[:, -24:]

                # 计算并记录验证阶段的各项指标
                target_numpy = targets.detach().cpu().numpy() * std + mean
                output_numpy = outputs.detach().cpu().numpy() * std + mean
                
                mae_vector.append(MAE(target_numpy, output_numpy))
                rmse_vector.append(RMSE(target_numpy, output_numpy))
                mape_vector.append(MAPE(target_numpy, output_numpy))
                smape_vector.append(sMAPE(target_numpy, output_numpy))

            # 将验证阶段的平均指标记录到 TensorBoard
            writer.add_scalar("MAE/val", np.mean(mae_vector), epoch)
            writer.add_scalar("RMSE/val", np.mean(rmse_vector), epoch)
            writer.add_scalar("MAPE/val", np.mean(mape_vector), epoch)
            writer.add_scalar("sMAPE/val", np.mean(smape_vector), epoch)

    # 训练结束后，使用 TorchScript 序列化并保存模型
    torch.jit.script(model).save(f"{save_path}")


# 禁用梯度计算装饰器，用于测试函数
@torch.no_grad()
def test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    test_data: DataLoader,
    load_path: str,
    compare_paths: List[str],
    save: bool,
    save_path: str,
    name: str,
) -> None:
    """
    该函数负责测试模型性能。

    参数:
        df_train: 包含训练数据的 DataFrame
        df_test: 包含测试数据的 DataFrame
        test_data: 测试数据的 DataLoader
        load_path: 待加载模型的路径
        compare_paths: 用于对比的其他模型结果路径列表
        save: 是否将当前模型保存为最佳模型
        save_path: 最佳模型的保存路径
        name: 保存模型时使用的名称
    """

    # 加载已保存的 TorchScript 模型
    model = torch.jit.load(f"{load_path}")

    # 将模型设置为评估模式
    model.eval()

    # 初始化用于存储真实值和预测值的张量
    real_values = torch.zeros((test_data.__len__(), 24))
    forecast = torch.zeros((test_data.__len__(), 24))

    # 遍历测试数据，模拟滚动预测过程
    i = 0
    for inputs_test in test_data:
        # 准备测试数据
        inputs_test = inputs_test.float()
        values = inputs_test[:, :-24, 0].unsqueeze(2)
        features = inputs_test[:, 24:, 1:3]
        targets = inputs_test[:, -24:, 0]

        # 动态更新训练集以反映实时预测场景：包含最近一年的历史和已发生的测试数据
        df_train = pd.concat([df_train[-365 * 24 :], df_test[: 24 * i]], axis=0)

        # 重新计算当前窗口的均值和标准差用于标准化
        means = {}
        stds = {}
        for variable in df_train.columns:
            means[variable] = df_train[variable].mean()
            stds[variable] = df_train[variable].std()

        # 对输入数据进行重新标准化
        values = (values - means["Price"]) / stds["Price"]
        features[:, :, 0] = (features[:, :, 0] - means["Exogenous 1"]) / stds[
            "Exogenous 1"
        ]
        features[:, :, 1] = (features[:, :, 1] - means["Exogenous 2"]) / stds[
            "Exogenous 2"
        ]

        # 计算模型预测输出
        outputs = model(values, features)[:, -24:]

        # 记录真实值和反标准化后的预测值
        real_values[i] = targets.detach().cpu()
        forecast[i] = outputs.detach().cpu() * stds["Price"] + means["Price"]

        # 增加步数计数
        i += 1

    # 将张量转换为 numpy 数组并计算总体评估指标
    real_values = real_values.numpy()
    forecast = forecast.numpy()
    mae = MAE(real_values, forecast)
    rmse = RMSE(real_values, forecast)
    mape = MAPE(real_values, forecast)
    smape = sMAPE(real_values, forecast)

    # 将预测结果保存为 CSV 文件
    df_forecast = pd.DataFrame(
        index=df_test.index[::24],
        columns=["h" + str(k) for k in range(24)],
        data=forecast,
    )
    if save:
        df_forecast.to_csv(f"{save_path}/forecast.csv")

    # 打印测试指标结果
    print(
        f"MAE: {mae:.3f}  |  RMSE: {rmse:.3f} | MAPE: {mape:.3f} | sMAPE: {smape:.3f}"
    )
    
    # 与其他基准模型进行 Diebold-Mariano (DM) 检验
    i = 1
    for compare_path in compare_paths:
        # 加载对比模型的预测结果
        forecast_dnn_ensemble = pd.read_csv(f"{compare_path}")
        forecast_dnn_ensemble = forecast_dnn_ensemble.to_numpy()[:, 1:]
        # 执行 DM 检验以确定预测差异是否显著
        dm_test = DM(
            real_values, forecast_dnn_ensemble, forecast, version="multivariate"
        )
        print(f"DM test {i}: {dm_test}")
        i += 1

    # 如果需要，将当前模型保存到指定位置
    if save:
        torch.jit.script(model).save(f"{save_path}/{name}.pt")

    return None
