"""
Plot tools
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_kupiec(hourly_LR_UC, critical_chi_square, max_LR_UC, title, horiz=24, plot_col=1):
    # hourly_LR_UC 是“模型 -> 24 个小时 LR_UC 统计量”的字典。
    plot_row = len(hourly_LR_UC) // plot_col
    x = np.arange(horiz)
    ks = list(hourly_LR_UC.keys())
    x_values = [0, horiz-1]
    y_values = [critical_chi_square, critical_chi_square]

    fig, axs = plt.subplots(plot_row, plot_col, constrained_layout=True, figsize=(2, 7.0))
    plt.subplots_adjust(hspace=0.01)
    #plt.title(title)
    for k, ax in zip(ks, axs.ravel()):
        # 每个子图画一个模型的 24 小时 Kupiec LR_UC，红线是卡方临界值。
        ax.plot(x, hourly_LR_UC[k], '^', color='navy')
        ax.set_ylim(bottom=-2, top=max_LR_UC + 2)
        #ax.set_title(k, fontsize=10)
        ax.set_ylabel("$LR_{UC}$")
        ax.plot(x_values, y_values, linestyle="--", color='firebrick')
        ax.grid()
        ax.label_outer()
    fig.suptitle(title, fontsize=10)
    fig.show()


def plot_quantiles(results: pd.DataFrame, target: str):
    # results 同时包含真实目标列和多个预测分位数列；蓝线是分位数，红线是真实值。
    title = target
    idx = results[target].index
    fig1, ax1 = plt.subplots()
    for i in results.columns.to_list():
        ax1.plot(idx, results[i], linestyle="-", color='steelblue', linewidth=0.9)

    ax1.plot(idx, results[target], '-', color='firebrick', label='$y_{true}$')
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel("$Price [EU/MWh]$")
    ax1.set_title(title)
    fig1.show()

