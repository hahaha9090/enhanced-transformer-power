# 导入深度学习相关的库
import torch
# 导入数值计算和数据处理库
import numpy as np
import pandas as pd
# 从 epftoolbox 库中导入数据读取、DNN 模型、数据构建和评估指标
from epftoolbox.data import read_data
from epftoolbox.models import DNN
from epftoolbox.models._dnn import _build_and_split_XYs
from epftoolbox.evaluation import MAE, RMSE, sMAPE, DM

# 导入系统操作和类型提示库
import os
from typing import Tuple, List, Dict, Literal

# 导入本项目定义的工具函数
from utils import load_data, forecast_next_day

# 设置计算设备：优先使用 GPU (cuda)，否则使用 CPU
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# 静态变量定义
DATASETS_PATH: str = "data" # 数据集存储路径
DATASETS_NAMES: Tuple[str, ...] = ("NP", "PJM", "BE", "FR", "DE") # 待测试的数据集名称列表


def main() -> None:
    """
    基准测试模块的主函数，负责根据配置执行不同的基准测试任务。

    异常:
        ValueError: 当基准测试名称无效时抛出。
    """

    # --- 变量配置 ---
    # 选择执行的基准测试类型：
    # - dnn_last_year: 仅使用最后一年数据标准化的 DNN
    # - dnn_all_past: 使用所有过去数据标准化的 DNN
    # - naive: 朴素模型（预测明天等于今天）
    # - final_results: 汇总最终结果
    benchmark: Literal[
        "dnn_last_year", "dnn_all_past", "naive", "final_results"
    ] = "final_results"

    # 根据配置执行对应的基准测试函数
    if benchmark == "dnn_last_year":
        benchmark_dnn_without_retrain(DATASETS_NAMES, True)
    elif benchmark == "dnn_all_past":
        benchmark_dnn_without_retrain(DATASETS_NAMES, False)
    elif benchmark == "naive":
        benchmark_naive_daily_model(DATASETS_NAMES)
    elif benchmark == "final_results":
        final_results(DATASETS_NAMES)
    else:
        raise ValueError("Invalid value for benchmark")


def benchmark_dnn_without_retrain(
    datasets: Tuple[str, ...], last_year: bool = False
) -> None:
    """
    执行不带重训练的 DNN 基准测试。

    参数:
        datasets: 使用的数据集名称列表。
        last_year: 布尔值，指示标准化是基于最后一年数据还是所有过去数据。
    """

    # 初始化结果列表
    results: List[List[float]] = []

    # 遍历每个数据集
    dataset: str
    for dataset in datasets:
        # 读取训练和测试数据
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        df_train, df_test = read_data(
            f"data/{dataset}",
            dataset=dataset,
            years_test=2,
            begin_test_date=None,
            end_test_date=None,
        )

        # 初始化 DNN 模型
        model: DNN = DNN(
            experiment_id="1",
            path_hyperparameter_folder="epftoolbox/examples/experimental_files",
            nlayers=2,
            dataset=dataset,
            years_test=2,
            shuffle_train=True,
            data_augmentation=0,
            calibration_window=4, # 校准窗口设为 4 年
        )

        # 准备预测结果的 DataFrame 容器
        forecast_dates = df_test.index[::24]
        forecast = pd.DataFrame(
            index=df_test.index[::24], columns=["h" + str(k) for k in range(24)]
        )
        # 准备真实值 DataFrame 容器
        real_values = df_test.loc[:, ["Price"]].values.reshape(-1, 24)
        real_values = pd.DataFrame(
            real_values, index=forecast.index, columns=forecast.columns
        )

        # 使用训练数据构建初始输入和输出特征
        Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = _build_and_split_XYs(
            dfTrain=df_train,
            features=model.best_hyperparameters,
            shuffle_train=True,
            dfTest=df_test,
            date_test=None,
            data_augmentation=model.data_augmentation,
            n_exogenous_inputs=len(df_train.columns) - 1,
        )

        # 对输入输出数据进行正则化
        Xtrain, Xval, Xtest, Ytrain, Yval = model._regularize_data(
            Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval
        )

        # 初始模型校准（重校准）
        model.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval)

        # 遍历测试集中的每个预测日期（按天滚动）
        for date in forecast_dates:
            # 准备当前日期可用的历史数据
            if last_year:
                # 仅考虑最近一年的数据
                data_available = pd.concat(
                    [
                        df_train[-365 * 24 :],
                        df_test.loc[: date + pd.Timedelta(hours=23), :],
                    ],
                    axis=0,
                )
            else:
                # 考虑所有过去的数据
                data_available = pd.concat(
                    [df_train, df_test.loc[: date + pd.Timedelta(hours=23), :]], axis=0
                )

            # 在可用数据中，将当前预测日期的价格设为 NaN（因为此时价格未知）
            data_available.loc[date : date + pd.Timedelta(hours=23), "Price"] = np.NaN

            # 调用接口进行次日电价预测
            Yp: np.ndarray = forecast_next_day(
                model, df=data_available, next_day_date=date
            )

            # 存储当天的预测结果
            forecast.loc[date, :] = Yp

        # 计算该数据集的总体评估指标
        mae: float = MAE(real_values.values, forecast.values)
        rmse: float = RMSE(real_values.values, forecast.values)
        smape: float = sMAPE(real_values.values, forecast.values)

        # 记录结果
        results.append([mae, rmse, smape])

        # --- 保存预测结果 ---
        # 确保保存目录存在
        save_dir = f"results/forecasts/dnn_ensemble_without_retraining_last_year_{last_year}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # 保存为 CSV
        forecast.to_csv(f"{save_dir}/{dataset}.csv")

    # --- 保存所有数据集的汇总结果 ---
    df: pd.DataFrame = pd.DataFrame(
        data=np.array(results), index=datasets, columns=["MAE", "RMSE", "sMAPE"]
    )
    if not os.path.isdir("results/benchmarks"):
        os.makedirs("results/benchmarks")
    df.to_csv(
        f"results/benchmarks/dnn_ensemble_without_retraining_last_year_{last_year}.csv"
    )

    return None


@torch.no_grad()
def benchmark_naive_daily_model(datasets: Tuple[str, ...]) -> None:
    """
    执行朴素模型基准测试：预测明天的电价等于今天的电价。

    参数:
        datasets: 待测试的数据集名称。
    """

    # 初始化指标结果
    results = []

    # 遍历每个数据集
    dataset: str
    for dataset in datasets:
        # 加载测试数据
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        df_train, df_test = read_data(
            f"{DATASETS_PATH}/{dataset}",
            dataset=dataset,
            years_test=2,
            begin_test_date=None,
            end_test_date=None,
        )
        # 拼接训练集最后一天的价格，以便预测测试集第一天
        df_test = pd.concat([df_train[-24:], df_test], axis=0)

        # 计算总天数
        test_number_of_days: int = len(df_test) // 24

        # 初始化真实值和预测值容器
        real_values: np.ndarray = np.zeros((test_number_of_days - 1, 24))
        forecast: np.ndarray = np.zeros((test_number_of_days - 1, 24))

        # 遍历每一天执行朴素预测
        for i in range(test_number_of_days - 1):
            # 真实值是“明天”
            real_values[i] = df_test["Price"][24 + 24 * i : 48 + 24 * i].to_numpy()
            # 预测值是“今天”
            forecast[i] = df_test["Price"][24 * i : 24 * i + 24].to_numpy()

        # 计算评估指标
        mae: float = MAE(real_values, forecast)
        rmse: float = RMSE(real_values, forecast)
        smape: float = sMAPE(real_values, forecast)

        # 记录结果
        results.append([mae, rmse, smape])

        # --- 保存预测结果 ---
        if not os.path.isdir(f"results/forecasts/naive_daily_model"):
            os.makedirs(f"results/forecasts/naive_daily_model")
        df_forecast = pd.DataFrame(
            index=df_test.index[24::24],
            columns=["h" + str(k) for k in range(24)],
            data=forecast,
        )
        df_forecast.to_csv(f"results/forecasts/naive_daily_model/{dataset}.csv")

    # 保存基准测试汇总结果
    df: pd.DataFrame = pd.DataFrame(
        data=np.round_(np.array(results), decimals=3),
        index=datasets,
        columns=["MAE", "RMSE", "sMAPE"],
    )
    if not os.path.isdir("results/benchmarks"):
        os.makedirs("results/benchmarks")
    df.to_csv(f"results/benchmarks/naive_daily_model.csv")

    return None


@torch.no_grad()
def final_results(datasets: Tuple[str, ...]) -> None:
    """
    汇总最终结果表，对比 Transformer 模型与其他基准模型的性能。

    参数:
        datasets: 待分析的数据集名称。
    """

    results = []

    for dataset in datasets:
        # 加载数据加载器
        train_data, val_data, test_data, mean, std = load_data(
            dataset, f"{DATASETS_PATH}/{dataset}", sequence_length=128, batch_size=128
        )

        # 加载本项目的最佳 Transformer 模型
        directories = os.listdir(f"best_models/{dataset}")
        # 筛选出模型文件路径
        model_path = (
            directories[1] if directories[0].split(".")[-1] == "csv" else directories[0]
        )
        model = torch.jit.load(f"best_models/{dataset}/{model_path}")

        # 将模型设为评估模式
        model.eval()

        # 初始化真实值容器
        real_values = torch.zeros((test_data.__len__(), 24))

        # 提取测试集真实目标值
        i = 0
        for inputs in test_data:
            inputs = inputs.float()
            targets = inputs[:, -24:, 0]
            real_values[i] = targets[0, -24:].detach().cpu()
            i += 1

        # 转换为 numpy 格式
        real_values = real_values.numpy()

        # 读取本项目的 Transformer 预测结果
        forecast = pd.read_csv(f"best_models/{dataset}/forecast.csv")
        forecast = forecast.to_numpy()[:, 1:]

        # 计算 Transformer 的核心指标
        mae = MAE(real_values, forecast)
        rmse = RMSE(real_values, forecast)
        smape = sMAPE(real_values, forecast)

        # --- 与其他基准模型进行 DM 检验 ---
        # 1. 对比 DNN 集成模型 (所有历史数据标准化)
        forecast_dnn_ensemble = pd.read_csv(
            f"results/forecasts/dnn_ensemble_without_retraining_last_year_False/{dataset}.csv"
        )
        forecast_dnn_ensemble = forecast_dnn_ensemble.to_numpy()[:, 1:]
        dm_test_dnn_ensemble = DM(
            real_values, forecast_dnn_ensemble, forecast, version="multivariate"
        )
        
        # 2. 对比 DNN 集成模型 (最近一年数据标准化)
        forecast_dnn_ensemble_last_year = pd.read_csv(
            f"results/forecasts/dnn_ensemble_without_retraining_last_year_True/{dataset}.csv"
        )
        forecast_dnn_ensemble_last_year = forecast_dnn_ensemble_last_year.to_numpy()[:, 1:]
        dm_test_dnn_ensemble_last_year = DM(
            real_values,
            forecast_dnn_ensemble_last_year,
            forecast,
            version="multivariate",
        )
        
        # 3. 对比朴素模型
        forecast_naive_model = pd.read_csv(
            f"results/forecasts/naive_daily_model/{dataset}.csv"
        )
        forecast_naive_model = forecast_naive_model.to_numpy()[:, 1:]
        dm_test_naive_model = DM(
            real_values, forecast_naive_model, forecast, version="multivariate"
        )

        # 收集所有指标
        results.append(
            [
                mae,
                rmse,
                smape,
                dm_test_dnn_ensemble,
                dm_test_dnn_ensemble_last_year,
                dm_test_naive_model,
            ]
        )

    # 将所有结果保存为最终的 CSV 表格
    df = pd.DataFrame(
        data=np.round_(np.array(results), decimals=3),
        index=datasets,
        columns=[
            "MAE",
            "RMSE",
            "sMAPE",
            "DM test DNN",
            "DM test DNN last year",
            "DM test Dum Model",
        ],
    )
    if not os.path.isdir(f"results"):
        os.makedirs(f"results")
    df.to_csv(f"results/results.csv")

    return None


# 程序入口点
if __name__ == "__main__":
    main()
