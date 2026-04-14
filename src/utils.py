# 导入深度学习相关的库
import torch
from torch.utils.data import Dataset, DataLoader
# 导入数值计算和数据处理库
import numpy as np
import pandas as pd
# 从 epftoolbox 库中导入数据读取和模型相关组件
from epftoolbox.data import read_data
from epftoolbox.models import DNN
from epftoolbox.models._dnn import _build_and_split_XYs

# 导入随机数、系统操作和类型提示库
import random
import os
from typing import Tuple, Dict

# 设置计算设备：优先使用 GPU (cuda)，否则使用 CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 定义电力数据集类，继承自 torch.utils.data.Dataset
class ElectricDataset(Dataset):
    def __init__(
        self, dataset: pd.DataFrame, sequence_length: int, evaluate: bool = False
    ) -> None:
        """
        ElectricDataset 的构造函数

        参数:
            dataset: DataFrame 格式的数据集。包含三列（价格、特征1、特征2），索引为时间格式。
            sequence_length: 每个样本提取的过去小时数。
            evaluate: 如果为 True，则仅在每天开始时（00:00）进行评估。
        """

        # 将 DataFrame 转换为 torch 张量并移动到指定设备
        self.dataset = torch.from_numpy(dataset.to_numpy()).to(device)
        
        # 提取时间特征：小时、星期几、月份，并作为额外的特征列
        dates = torch.zeros((len(dataset), 3)).to(device)
        dates[:, 0] = torch.Tensor(dataset.index.hour.to_numpy())
        dates[:, 1] = torch.Tensor(dataset.index.dayofweek.to_numpy())
        dates[:, 2] = torch.Tensor(dataset.index.month.to_numpy())
        
        # 将原始数据与时间特征在特征维度上拼接
        self.dataset = torch.concat((self.dataset, dates), dim=1)
        self.sequence_length = sequence_length
        self.evaluate = evaluate

    def __len__(self) -> int:
        """
        返回数据集的长度（样本数量）。

        返回:
            数据集中有效样本的数量。
        """
        if self.evaluate:
            # 评估模式下，按天（24小时）计算样本数
            return (self.dataset.size(0) - self.sequence_length) // 24
        else:
            # 训练模式下，滑动窗口计算样本数
            return self.dataset.size(0) - self.sequence_length - 23

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        根据索引返回数据集中的一个样本。

        参数:
            index: 元素的索引。

        返回:
            包含过去 sequence_length 小时及接下来 24 小时的数据张量。
        """

        # 根据索引提取对应窗口的数据
        if self.evaluate:
            # 评估模式：跳转到每一天的起始位置提取
            values = self.dataset[24 * index : 24 * index + self.sequence_length + 24]
        else:
            # 训练模式：连续滑动提取
            values = self.dataset[index : index + self.sequence_length + 24]

        return values


def load_data(
    dataset_option: str,
    save_path: str,
    sequence_length: int = 72,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    该方法返回所选数据集的 DataLoader。

    参数:
        dataset_option: 数据集名称。
        save_path: 数据保存路径。
        sequence_length: 神经网络的输入序列长度，必须是 24 的倍数。
        batch_size: 批处理大小。
        shuffle: 是否打乱数据。
        drop_last: 是否丢弃最后一个不完整的批次。
        num_workers: 加载数据的工作线程数。

    返回:
        训练、验证和测试集的 DataLoader，以及价格的均值和标准差。
    """

    # 如果保存路径不存在，则创建目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 从远程仓库下载数据集（CSV 格式）
    url_dir: str = "https://zenodo.org/records/4624805/files/"
    data: pd.DataFrame = pd.read_csv(url_dir + dataset_option + ".csv", index_col=0)
    # 将下载的数据保存到本地
    file_path: str = os.path.join(save_path, dataset_option + ".csv")
    data.to_csv(file_path)

    # 读取本地文件，本地数据集路径：data/[dataset]/[dataset].csv
    ## local_path = os.path.join(path, dataset_option + ".csv")  # path是load_data的入参（即DATASETS_PATH/dataset）
    ## data: pd.DataFrame = pd.read_csv(local_path, index_col=0)



    # 读取并划分数据集（训练/验证集与测试集）
    df_train_val: pd.DataFrame
    df_test: pd.DataFrame
    df_train_val, df_test = read_data(
        dataset=dataset_option, path=save_path, years_test=2
    )
    # 将训练集末尾的数据拼接到测试集开头，以保证测试集第一个样本有足够的历史序列
    df_test = pd.concat([df_train_val[-sequence_length:], df_test], axis=0)
    
    # 计算训练验证集的均值和标准差，并进行 Z-score 标准化
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for variable in df_train_val.columns:
        means[variable] = df_train_val[variable].mean()
        stds[variable] = df_train_val[variable].std()
        df_train_val[variable] = (df_train_val[variable] - means[variable]) / stds[
            variable
        ]

    # 进一步划分训练集和验证集（验证集取最后 42 周的数据）
    df_train: pd.DataFrame = df_train_val[: -42 * 7 * 24]
    df_val: pd.DataFrame = df_train_val[-42 * 7 * 24 :]
    # 同样为验证集开头补充历史序列
    df_val = pd.concat([df_train[-sequence_length:], df_val], axis=0)
    
    # 创建 Dataset 对象
    train_dataset: Dataset = ElectricDataset(df_train, sequence_length)
    val_dataset: Dataset = ElectricDataset(df_val, sequence_length, evaluate=True)
    test_dataset: Dataset = ElectricDataset(df_test, sequence_length, evaluate=True)

    # 初始化 DataLoader
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=1, # 测试时通常按天（batch=1）预测
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    # 返回加载器及用于反标准化的统计量
    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        means["Price"],
        stds["Price"],
    )


def set_seed(seed: int) -> None:
    """
    该函数设置随机种子，确保实验过程具有确定性。

    参数:
        seed: 随机种子值。
    """

    # 设置 NumPy 和 Python 内置随机库的种子
    np.random.seed(seed)
    random.seed(seed)

    # 设置 PyTorch 的种子并启用确定性算法
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # 确保 GPU 上的所有操作也是确定性的
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 针对 CUDA >= 10.2 的确定性行为配置环境变量
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def forecast_next_day(model: DNN, df: pd.DataFrame, next_day_date) -> np.ndarray:
    """
    为 DNN 模型的每日重校准和预测提供简单易用的接口。

    该方法接收历史数据 df 和目标预测日期 next_day_date。
    它会使用直到目标日期前一天的数据重新训练模型，并预测目标日期的电价。

    参数:
        df: 历史数据 DataFrame，包含价格和 N 个外部输入。索引为小时频率的日期。
        next_day_date: 目标预测日期。

    返回:
        包含指定日期预测值的数组。
    """

    # 根据 calibration_window 定义新的训练数据集（考虑过去若干年的数据）
    df_train = df.loc[: next_day_date - pd.Timedelta(hours=1)]
    df_train = df_train.loc[
        next_day_date - pd.Timedelta(hours=model.calibration_window * 364 * 24) :
    ]

    # 定义测试数据集：包含目标日期及前两周的数据，以便构建输入特征
    df_test = df.loc[next_day_date - pd.Timedelta(weeks=2) :, :]

    # 生成训练、验证和测试集的输入输出
    # 即使 df_test 包含 15 天数据，通过 date_test 参数确保 Xtest 仅反映目标日期的特征
    Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = _build_and_split_XYs(
        dfTrain=df_train,
        features=model.best_hyperparameters,
        shuffle_train=True,
        dfTest=df_test,
        date_test=next_day_date,
        data_augmentation=model.data_augmentation,
        n_exogenous_inputs=len(df_train.columns) - 1,
    )

    # 如果需要，对输入输出进行正则化/标准化处理
    Xtrain, Xval, Xtest, Ytrain, Yval = model._regularize_data(
        Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval
    )

    # 重新校准神经网络并提取预测结果
    Yp = model.predict(Xtest=Xtest)

    return Yp
