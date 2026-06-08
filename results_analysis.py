"""
Script to plot experiments results
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

from tools.results_analysis_tools import ExperAnalysis

#----------------------------------------------------------------------------------------------------------------
# Select PF task
#----------------------------------------------------------------------------------------------------------------
# 这个脚本用于交互式查看单个市场的聚合结果：画 Kupiec、PICP、DM 检验图，并打印 KPI 表。
# 要分析其他市场，只需要改 PF_task；run_id 要与 exec_qra_cp.py 保存的聚合结果一致。
PF_task = 'MGP_NORD'
run_id = 'recalib_opt_grid_1'

#----------------------------------------------------------------------------------------------------------------
# Load experiments results and create the results analysis object
#----------------------------------------------------------------------------------------------------------------
exper_results = ExperAnalysis(PF_task=PF_task, run_id=run_id)

# ----------------------------------------------------------------------------------------------------------------
# Plot kupiec test
# ----------------------------------------------------------------------------------------------------------------
exper_results.plot_kupiec()

# ----------------------------------------------------------------------------------------------------------------
# Plot hourly PICP
# ----------------------------------------------------------------------------------------------------------------
# select the subset to plot
# 这里保留多组可选配置，实际绘图时只启用当前列表中未注释的模型。
conf_to_plot=[
               #'QRA-DNN','CP-DNN','N-DNN','JSU-DNN','STU-DNN','QR-DNN',
               #'QRA-DNN','CQ-N-DNN', 'CQ-JSU-DNN', 'CQ-STU-DNN','CQ-QR-DNN',
               'QRA-DNN', 'OCQ-N-DNN', 'OCQ-JSU-DNN', 'OCQ-STU-DNN', 'OCQ-QR-DNN'
]
exper_results.plot_stepwise_PICP(conf_to_plot)

#----------------------------------------------------------------------------------------------------------------
# Execute DM test on pinball and winkler's score
#----------------------------------------------------------------------------------------------------------------
exper_results.plot_DM_test_pinball()
exper_results.plot_DM_test_winkler()
exper_results.plot_DM_test_mae()

#----------------------------------------------------------------------------------------------------------------
# Plot test Preds
#----------------------------------------------------------------------------------------------------------------
# 画某一个方法的预测分位数曲线，并叠加真实电价序列。
exper_results.plot_experiment_predictions('OCQR', PF_task)

# ----------------------------------------------------------------------------------------------------------------
# Print latex table of mean KPIs
# ----------------------------------------------------------------------------------------------------------------
# 分别输出论文可用的 LaTeX 表和便于阅读的 Markdown 表。
print(exper_results.table_mean_kpis_latex())
print(exper_results.table_mean_kpis_markdown())
