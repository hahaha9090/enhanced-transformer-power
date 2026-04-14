# 导入深度学习相关的库
import torch
import numpy as np  # 新增：导入numpy
from torch.utils.tensorboard import SummaryWriter
# 从 epftoolbox 库中导入读取数据的函数
from epftoolbox.data import read_data
# 导入 Adafactor 优化器及其调度器
from transformers.optimization import Adafactor, AdafactorSchedule

# 导入系统操作、类型提示等标准库
import os
from typing import Optional, Literal, List

# 导入本项目定义的工具函数和模型
from utils import set_seed, load_data
from models import BaseDailyElectricTransformer, sMAPELoss
from train_functions import train, test

# ========== 1. 正确定义GPU设备 ==========
device: torch.device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # 指定cuda:0（单GPU）
)

# 设置全局随机种子以确保实验的可复现性
set_seed(42)

# 定义数据集存放的基础路径
DATASETS_PATH: str = "data"


def main() -> None:
    """
    训练模块的主函数，负责配置超参数、加载数据、初始化模型、定义损失函数和优化器，并执行训练或测试任务。

    异常:
        ValueError: 当损失函数、优化器、调度器或执行模式名称无效时抛出。
    """

    # --- 变量配置 ---
    dataset: Literal["NP", "PJM", "FR", "BE", "DE"] = "FR"
    exec_mode: Literal["train", "test"] = "train"
    save_model = True

    # --- 超参数设置 ---
    epochs = 2                   # 训练轮数，原定100
    sequence_length = 336        # 输入序列长度（小时数，336小时=14天）
    lr = 1e-4                    # 学习率
    num_layers = 4               # Transformer 层数
    num_heads = 4                # 注意力头数
    embedding_dim = 128          # 嵌入维度
    dim_feedforward = 1048       # 前馈网络维度
    normalize_first = False      # 是否在 Transformer 层中先进行归一化
    dropout = 0.2                # 随机失活率
    activation = "relu"          # 激活函数类型
    loss_name = "mae"            # 损失函数类型：mae, mse, smooth_mae, sMAPE
    optimizer_name = "adamw"     # 优化器类型：adamw, adam, sgd, rmsprop 等
    weight_decay = 1e-2          # 权重衰减率
    clip_gradients = 0.1         # 梯度裁剪阈值
    scheduler_name = "steplr_70_0.1" # 学习率调度器配置

    # ========== 2. 打印GPU详细信息（验证） ==========
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")

    # --- 数据加载 ---
    train_dataloader, val_dataloader, test_dataloader, mean, std = load_data(
        dataset,
        f"{DATASETS_PATH}/{dataset}",
        sequence_length=sequence_length,
        batch_size=32,  # 3. 降低batch_size（MX450显存较小，避免OOM）原本128
    )

    # ========== 4. 统一mean/std为GPU张量+保留NumPy版本 ==========
    # GPU张量（用于模型计算）
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device)
    # NumPy版本（用于train_functions中的数值计算）
    mean_np = np.array(mean)
    std_np = np.array(std)

    # 使用 epftoolbox 读取原始数据
    df_train, df_test = read_data(
        dataset=dataset, path=f"{DATASETS_PATH}/{dataset}", years_test=2
    )
    means = {}
    stds = {}
    for variable in df_train.columns:
        means[variable] = df_train[variable].mean()
        stds[variable] = df_train[variable].std()

    # --- 日志与模型定义 ---
    name: str = (
        f"m_ed_{embedding_dim}_nh_{num_heads}_df_{dim_feedforward}_nl_{num_layers}_sl_"
        f"{sequence_length}_{normalize_first}_d_{dropout}_a_{activation}_l_{loss_name}_o_{optimizer_name}_"
        f"lr_{lr}_wd_{weight_decay}_cg_{clip_gradients}_s_{scheduler_name}_e_{epochs}"
    )
    writer: SummaryWriter = SummaryWriter(f"runs/{dataset}/{name}")

    # ========== 5. 模型移到GPU ==========
    model: torch.nn.Module = BaseDailyElectricTransformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        normalize_first=normalize_first,
        dropout=dropout,
        activation=activation,
    ).to(device)

    # --- 定义损失函数 ---
    loss: torch.nn.Module
    if loss_name == "mse":
        loss = torch.nn.MSELoss()
    elif loss_name == "mae":
        loss = torch.nn.L1Loss()
    elif loss_name == "smooth_mae":
        loss = torch.nn.SmoothL1Loss()
    elif loss_name == "sMAPE":
        loss = sMAPELoss()
    else:
        raise ValueError("Invalid loss name")

    # ========== 6. 损失函数移到GPU ==========
    loss = loss.to(device)

    # --- 定义优化器 ---
    optimizer: torch.optim.Optimizer
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "Adafactor_no_warmup":
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=lr,
            weight_decay=weight_decay,
            clip_threshold=clip_gradients,
        )
    elif optimizer_name == "Adafactor":
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=lr,
            weight_decay=weight_decay,
            clip_threshold=clip_gradients,
        )
    else:
        raise ValueError("Invalid optimizer name")

    # --- 定义学习率调度器 ---
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    if scheduler_name is not None:
        scheduler_name_pieces = scheduler_name.split("_")
        if scheduler_name_pieces[0] == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_name_pieces[1]),
                gamma=float(scheduler_name_pieces[2]),
            )
        elif scheduler_name_pieces[0] == "multisteplr":
            milestones = [int(piece) for piece in scheduler_name_pieces[1:-1]]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=float(scheduler_name_pieces[-1])
            )
        elif scheduler_name_pieces[0] == "adafactorschedule":
            scheduler = AdafactorSchedule(optimizer)
        else:
            raise ValueError("Invalid scheduler name")
    else:
        scheduler = None

    # --- 目录与路径准备 ---
    if not os.path.isdir(f"models/{dataset}"):
        os.makedirs(f"models/{dataset}")
    if not os.path.isdir(f"best_models/{dataset}"):
        os.makedirs(f"best_models/{dataset}")

    model_path: str = f"models/{dataset}/{name}.pt"

    compare_paths: List[str] = [
        f"results/forecasts/dnn_ensemble_without_retraining_last_year_False/{dataset}.csv",
        f"results/forecasts/naive_daily_model/{dataset}.csv",
    ]

    # --- 执行模式分支 ---
    if exec_mode == "train":
        # ========== 7. 传入NumPy版本的mean/std ==========
        train(
            train_dataloader,
            val_dataloader,
            model,
            mean_np,  # 改为NumPy版本
            std_np,   # 改为NumPy版本
            loss,
            optimizer,
            scheduler,
            clip_gradients,
            epochs,
            writer,
            model_path,
        )

    elif exec_mode == "test":
        if not os.path.isdir(f"best_models/{dataset}"):
            os.makedirs(f"best_models/{dataset}")

        if save_model:
            files = os.listdir(f"best_models/{dataset}")
            if len(files) != 0:
                for file in files:
                    os.remove(f"best_models/{dataset}/{file}")

        best_model_path: str = f"best_models/{dataset}"

        test(
            df_train,
            df_test,
            test_dataloader,
            model_path,
            compare_paths,
            save_model,
            best_model_path,
            name,
        )

    else:
        raise ValueError("Invalid execution mode")


if __name__ == "__main__":
    main()