"""
Script to run the recalibration experiments
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs


#--------------------------------------------------------------------------------------------------------------------
# Run recalibration configs
#--------------------------------------------------------------------------------------------------------------------
# 这个脚本是“从头重新训练 + 滚动重新校准”的主入口。
# 执行成本很高：下面的配置会决定跑哪个电力市场、哪些模型、哪些 ensemble 子实验。
# Set PEPF task to execute
PF_task_name = 'DE_price'
# List of models setup to execute
setups_to_experiment = ['point-DNN', 'QR-DNN', 'JSU-DNN', 'STU-DNN', 'N-DNN']
# List of runs id (e.g., for ensemble components)
runs_id = ['recalib_opt_grid_1_1', 'recalib_opt_grid_1_2','recalib_opt_grid_1_3','recalib_opt_grid_1_4']
# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'

#---------------------------------------------------------------------------------------------------------------------
# 外层按模型类型循环，内层按 ensemble 组件循环；每个组合都有自己的 exper_configs.json 和结果目录。
for exper_setup in setups_to_experiment:
    for run_id in runs_id:
        # Load experiments configuration from json file
        configs=load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

        # Instantiate recalibratione engine
        # 引擎内部负责：加载数据、构造滑动窗口样本、训练模型、逐个测试日重新校准并保存预测结果。
        PrTSF_eng = PrTsfRecalibEngine(data_configs=configs['data_config'],
                                       model_configs=configs['model_config'])

        # Exec recalib loop over the test_set samples, using the tuned hyperparams
        # hyper_mode='load_tuned' 时直接读取已调好的超参数；设为 optuna_tuner 才会重新搜索超参数。
        test_predictions = PrTSF_eng.run_recalibration(hyper_mode=hyper_mode)
