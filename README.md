# PefCodeBench

本代码库用于复现论文（预印本，审稿中）中的实验：

- Brusaferri, A., Ballarino, A., Grossi, L., & Laurini, F. (2024). 在线共形化神经网络集成用于日前电价概率预测（On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices），https://arxiv.org/abs/2404.02722

---

### 快速开始

实验结果以 pickle 文件形式存储在 <code>experiments/task</code> 文件夹中，
并按照 <code>区域 -> 方法 -> 重新校准</code> 的运行结果进行聚合。

注意：在代码中，我们使用 <code>DE</code> 表示德国电力市场；而在论文中使用的是 <code>GE</code>，因为 <code>DE</code> 已经用于表示 Deep Ensembles。

所使用的软件包版本记录在 <code>requirements.txt</code> 文件中（Python 3.8.10）。此外，DM 检验、Kupiec 检验、分布式神经网络（Distributional NNs）和共形预测区间（Conformal PI）的相关代码分别基于以下项目构建：https://github.com/jeslago/epftoolbox、https://github.com/rafa-rod/vartests、https://github.com/gmarcjasz/distributionalnn 和 https://github.com/aangelopoulos/conformal-time-series。

<code>results_analysis.py</code> 脚本包含从已存储的 pickle 文件中生成论文 *Results* 部分图表的函数。

如果需要从头执行重新校准实验（即重新训练模型），请打开 <code>run_recalibration.py</code> 脚本，
在 <code>PF_task_name</code> 变量中选择要执行的数据集，然后运行该脚本。
训练算法到达的具体局部最小值可能会导致测试预测结果出现波动（例如 QR、JSU 和 Stu 之间的差异）。
不过，在不同实验设置下，已实现的基于共形预测的技术预计能够改善骨干模型的逐小时校准效果。

<code>run_recalibration.py</code> 脚本会将实验结果存储到对应的 <code>results</code> 子文件夹中。
在运行实验之前，请通过定义一个新名称（例如 <code>my_recalib_opt_grid_1_(1-4)</code>）创建 <code>recalib_opt_grid_1_(1-4)</code> 文件夹的副本，
以保留原始实验结果；否则，每次运行脚本时这些结果都会被更新。

请将 <code>hyper_mode</code> 变量保持为 <code>'load_tuned'</code>，以加载已存储的超参数取值。
如果还需要从头执行超参数搜索，请将其设置为 <code>'optuna_tuner'</code>。

重新校准运行完成后，请运行 <code>exec_qra_cp.py</code> 脚本，
以执行后处理流程，即分位数回归平均（Quantile Regression Averaging）和共形预测（Conformal Prediction）。

如果你创建了自己的实验副本，请在 <code>run_recalibration.py</code>、<code>exec_qra_cp.py</code> 和 <code>results_analysis.py</code> 中，将所选择的名称赋给 <code>run_id</code> 变量。

---

### 引用

如果你在自己的论文中使用了本代码，请引用我们的论文：
https://arxiv.org/abs/2404.02722

      @misc{brusaferri2024online,
      title={On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices},
      author={Alessandro Brusaferri and Andrea Ballarino and Luigi Grossi and Fabrizio Laurini},
      year={2024},
      eprint={2404.02722},
      archivePrefix={arXiv},
      primaryClass={cs.LG}}
