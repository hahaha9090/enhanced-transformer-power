# 导入深度学习相关的库
import torch
import numpy as np
import pandas as pd

# 导入随机数、系统操作、绘图和类型提示库
import random
import os
import matplotlib.pyplot as plt
from typing import Tuple

# 导入本项目定义的工具函数
from utils import set_seed

# 设置全局随机种子以确保绘图结果的可复现性
set_seed(42)

# 静态变量定义
FORECASTS_PATH: str = "results/forecasts" # 预测结果存储的基础路径
DATASETS_NAMES: Tuple[str, ...] = ("NP", "PJM", "BE", "FR", "DE") # 数据集名称列表


def main() -> None:
    """
    可视化模块的主入口函数，用于生成预测对比图。
    """

    # 调用 draw_forecasts 函数为所有数据集生成可视化图表，并保存到 "visualizations" 目录
    draw_forecasts(DATASETS_NAMES, "visualizations")


def draw_forecasts(
    datasets: Tuple[str, ...], save_path: str, number_per_day: int = 2
) -> None:
    """
    该函数创建并保存不同模型预测结果与真实值的对比可视化图表。

    参数:
        datasets: 数据集名称的元组。
        save_path: 可视化图表的保存目录。
        number_per_day: 每天生成的可视化图表数量（目前代码中未完全使用该参数逻辑）。
    """

    for dataset in datasets:
        # --- 加载各个模型的预测结果 ---
        # 1. 加载朴素模型（通常作为真实值的参考）的结果
        real_values_df: pd.DataFrame = pd.read_csv(
            f"{FORECASTS_PATH}/naive_daily_model/{dataset}.csv"
        )
        # 2. 加载仅使用最后一年数据训练的 DNN 集成模型结果
        dnn_last_year_df: pd.DataFrame = pd.read_csv(
            f"{FORECASTS_PATH}/dnn_ensemble_without_retraining_last_year_True/{dataset}.csv"
        )
        # 3. 加载使用所有年份数据训练的 DNN 集成模型结果
        dnn_all_years_df: pd.DataFrame = pd.read_csv(
            f"{FORECASTS_PATH}/dnn_ensemble_without_retraining_last_year_False/{dataset}.csv"
        )
        # 4. 加载本项目 Transformer 模型的最佳预测结果
        transformers_df: pd.DataFrame = pd.read_csv(
            f"best_models/{dataset}/forecast.csv"
        )

        # 随机选择一个索引，用于展示某一天或某一段的预测效果
        index = random.randint(1, dnn_last_year_df.shape[1])

        # --- 提取对应的数值数组 ---
        # 提取真实值（从朴素模型文件中获取）
        real_values: np.ndarray = real_values_df.to_numpy()[index + 1, 1:]
        # 提取各个模型的预测值
        dnn_last_year_results: np.ndarray = dnn_last_year_df.to_numpy()[index, 1:]
        dnn_all_years_results: np.ndarray = dnn_all_years_df.to_numpy()[index, 1:]
        transformers_results: np.ndarray = transformers_df.to_numpy()[index, 1:]

        # --- 创建绘图 ---
        plt.figure()
        # 绘制 24 小时的曲线对比
        plt.plot(np.arange(24), real_values)           # 真实值
        plt.plot(np.arange(24), transformers_results) # Transformer 预测
        plt.plot(np.arange(24), dnn_last_year_results) # DNN 集成 1 预测
        plt.plot(np.arange(24), dnn_all_years_results) # DNN 集成 2 预测
        
        # 设置图表标签和图例
        plt.xlabel("time [hour]")
        plt.ylabel("Price [EUR/MWh]")
        plt.legend(["Real values", "Transformer", "DNN Ensemble 1", "DNN Ensemble 2"])

        # --- 保存并关闭图表 ---
        # 如果保存目录不存在，则创建
        if not os.path.isdir(f"{save_path}"):
            os.makedirs(f"{save_path}")
        # 将图表保存为 PDF 文件
        plt.savefig(f"{save_path}/{dataset}.pdf")
        # 关闭当前图形窗口以释放内存
        plt.close()

    return None


# 程序入口点
if __name__ == "__main__":
    main()
