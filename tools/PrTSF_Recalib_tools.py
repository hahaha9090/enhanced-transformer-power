"""
Main tools managing the recalibration process
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import os
import sys
from datetime import date
from datetime import datetime
from typing import Dict, List, Union
import re
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
import tensorflow as tf
from tools.prediction_quantiles_tools import plot_quantiles
from tools.email_box import send_experimentcompleted_email

from tools.data_utils import columns_keys, features_keys
from tools.models.models_tools import regression_model, Ensemble, get_model_class_from_conf


class RecalibBlock:
    """
    Class used to structure train/vali samples related to each recalibration block
    """
    def __init__(self, x_train, y_train, x_vali, y_vali):
        # 一个 recalibration block 保存当前测试样本对应的训练集和验证集。
        self.x_train = x_train
        self.y_train = y_train
        self.x_vali = x_vali
        self.y_vali = y_vali


class RecalibSamples:
    """
    Class used to structure the recalibration samples, including the test sample and the related list of recalib blocks
    """
    def __init__(self, x_test, y_test):
        # x_test/y_test 是当前滚动测试点；recalibBlocks 是可用于训练/验证的样本块列表。
        self.x_test = x_test
        self.y_test = y_test
        self.recalibBlocks = []

    def add_recal_block(self,x_train, y_train, x_vali, y_vali):
        # 当前实现通常只添加一个训练/验证块，但用列表保留扩展多个块的能力。
        self.recalibBlocks.append(RecalibBlock(x_train=x_train, y_train=y_train,
                                               x_vali=x_vali, y_vali=y_vali))


class WindowGenerator:
    """
    Creates the shifting windows, following the approach reported in the TF docs
    """
    def __init__(self,
                 input_width: int,
                 label_width: int,
                 shift: int,
                 data_columns: List,
                 target_columns: List = None):

        # Work out the label column indices.
        # label_columns 是目标变量列；column_indices 记录原始特征列在窗口张量最后一维中的位置。
        self.label_columns = target_columns
        if target_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(target_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(data_columns)}

        # Store the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # create slice object
        # To include the future conditioning features, the input slide include the prediction steps
        # The target column MUST be removed during input conditioning construction
        # 输入窗口包含历史滞后段和未来已知特征段；真正的标签只取最后 label_width 个小时。
        self.input_slice = slice(0, input_width + label_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        # features 形状通常是 [样本数, 总窗口长度, 全部列数]。
        inputs = features[:, self.input_slice, :]
        targets = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            # 只抽取目标列作为 y，避免把其他协变量误当成标签。
            targets = np.stack(
                [targets[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        return inputs, np.squeeze(targets, axis=-1)


class PrTsfDataloaderConfigs:
    """
    Class used to handle the dataloader configuration
    """
    def __init__(self,
                 task_name: str,
                 exper_setup: str,
                 dataset_name:str,
                 idx_start_train: Union[int, date],
                 idx_start_oos_preds: Union[int, date],
                 idx_end_oos_preds: Union[int, date],
                 num_vali_samples: int=180,
                 steps_lag_win: int=7,
                 pred_horiz: int=24,
                 preprocess: str='StandardScaler',
                 keep_past_train_samples: bool=True,
                 shuffle_mode: str='none'
                 ):
        # 这个配置对象集中保存一个市场/模型实验的数据切片方式和窗口长度。
        self.task_name = task_name
        self.exper_setup = exper_setup
        self.dataset_name = dataset_name
        self.idx_start_train = idx_start_train
        self.idx_start_oos_preds = idx_start_oos_preds
        self.idx_end_oos_preds = idx_end_oos_preds
        self.num_vali_samples = num_vali_samples
        self.steps_lag_win = steps_lag_win
        self.pred_horiz = pred_horiz
        self.preprocess = preprocess
        self.keep_past_train_samples = keep_past_train_samples
        self.shuffle_mode = shuffle_mode


#
def load_data_model_configs(task_name: str, exper_setup: str, run_id: str):
    """
    Load experiment configurations from json and build the handler object
    """
    path = os.path.join(os.getcwd(), 'experiments', 'tasks', task_name, exper_setup, run_id,'exper_configs.json')
    # Load experiment settings from json
    # 每个 task/model/run_id 目录下的 exper_configs.json 同时包含 data_config 和 model_config。
    with open(path) as f:
        expe_confs = json.load(f)

    # JSON 中日期被拆成 y/m/d 字段；这里恢复成 Python date，便于后续按日期定位全局索引。
    expe_confs['data_config']['idx_start_train'] = date(year=expe_confs['data_config']['idx_start_train']['y'],
                                                        month=expe_confs['data_config']['idx_start_train']['m'],
                                                        day=expe_confs['data_config']['idx_start_train']['d'])
    expe_confs['data_config']['idx_start_oos_preds'] = date(year=expe_confs['data_config']['idx_start_oos_preds']['y'],
                                                            month=expe_confs['data_config']['idx_start_oos_preds']['m'],
                                                            day=expe_confs['data_config']['idx_start_oos_preds']['d'])
    expe_confs['data_config']['idx_end_oos_preds'] = date(year=expe_confs['data_config']['idx_end_oos_preds']['y'],
                                                          month=expe_confs['data_config']['idx_end_oos_preds']['m'],
                                                          day=expe_confs['data_config']['idx_end_oos_preds']['d'])
    # Store exper run id
    # run_id 同时也是结果保存目录的一部分，必须写回 model_config 供引擎使用。
    expe_confs['model_config']['run_id'] = run_id

    # Append running experiments configs
    # 用强字段对象承载 data_config，减少后续主流程里的字典键拼写风险。
    data_configs = PrTsfDataloaderConfigs(
        task_name=task_name,
        exper_setup=exper_setup,
        dataset_name=expe_confs['data_config']['dataset_name'],
        idx_start_train=expe_confs['data_config']['idx_start_train'],
        idx_start_oos_preds=expe_confs['data_config']['idx_start_oos_preds'],
        idx_end_oos_preds=expe_confs['data_config']['idx_end_oos_preds'],
        num_vali_samples=expe_confs['data_config']['num_vali_samples'],
        steps_lag_win=expe_confs['data_config']['steps_lag_win'],
        pred_horiz=expe_confs['data_config']['pred_horiz'],
        preprocess=expe_confs['data_config']['preprocess'],
        keep_past_train_samples=expe_confs['data_config']['keep_past_train_samples'],
        shuffle_mode=expe_confs['data_config']['shuffle_mode'],
    )
    return {'data_config': data_configs,
            'model_config': expe_confs['model_config']}


class PrTsfRecalibEngine:
    """
    Main class executing the recalibration process
    """
    def __init__(self, data_configs: PrTsfDataloaderConfigs,
                 model_configs: Dict):

        self.data_configs = data_configs
        # load dataset csv file
        # 数据 CSV 是完整市场时间序列，后续会按配置截取训练/测试时间段。
        self.dataset = self.__load_dataset_from_file__(dataset_name=data_configs.dataset_name)
        # store the samples involved in the configured experimental period (between start_train and oos_end) and reindex
        self.__store_reindexed_dataset__(data_configs=data_configs)
        # build test samples idxs used by the recalibration iterator
        # test_set_idxs 保存每个日前测试样本的起始行索引，步长为 pred_horiz。
        self.test_set_idxs = self.__build_test_samples_idxs__()
        # instantiate preprocessing_objs
        self.preproc = self.__instantiate_preproc__()

        # store model configs and add internal confs automatically
        self.model_configs = model_configs
        # 根据配置字符串选择模型类，目前实现只有 DNNRegressor。
        self.model_class = get_model_class_from_conf(self.model_configs['model_class'])
        # Copy pred_horizon from data confs
        self.model_configs['pred_horiz'] = self.data_configs.pred_horiz
        # Build target quantiles from alpha, including the median
        self.model_configs['target_quantiles'] = self.__build_target_quantiles__(self.model_configs['target_alpha'])
        # Build maping between quantile idx and alpha/median
        # q_alpha_map 后面用于从预测分位数矩阵里快速找到某个 alpha 的上下界列。
        self.model_configs['q_alpha_map'] = self.__build_alpha_quantiles_map__(
                                                            target_quantiles=self.model_configs['target_quantiles'],
                                                            target_alpha=self.model_configs['target_alpha'])

    @staticmethod
    def __load_dataset_from_file__(dataset_name: str):
        """
        Load data from csv
        """
        dir_path = os.getcwd()
        # CSV 第一列被设为索引，但真正用于模型切片的是后续重建的全局行号。
        ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', dataset_name))
        ds.set_index(ds.columns[0], inplace=True)
        return ds


    def __get_global_idx_from_date__(self, date_id, mode='start'):
        """
        Get the global idx related to the input date.
        Mode: 'start': return the idx of first sub_step; 'end': return the idx of first sub_step
        """
        # 同一天有多个小时/子步，start 取当天第一行，end 取当天最后一行。
        date_idxs= self.dataset[self.dataset[columns_keys['Date']]== date_id.strftime('%Y-%m-%d')].index.tolist()
        if mode=='start':
            global_idx = date_idxs[0]
        elif mode=='end':
            global_idx = date_idxs[-1]
        else:
            sys.exit('ERROR: selected mode do not exist')

        return global_idx

    def __store_reindexed_dataset__(self, data_configs: PrTsfDataloaderConfigs):
        """
        Get train/test periods from configs and store
        """
        if (type(data_configs.idx_start_train) is date and
                type(data_configs.idx_start_oos_preds) is date and
                type(data_configs.idx_end_oos_preds) is date):
            self.data_configs = data_configs
            # set idx from input date
            # 配置用日期表达时，先映射成原始 CSV 中的全局行号。
            self.data_configs.idx_start_train = self.__get_global_idx_from_date__(self.data_configs.idx_start_train, mode='start')
            self.data_configs.idx_start_oos_preds = self.__get_global_idx_from_date__(self.data_configs.idx_start_oos_preds, mode='start')
            self.data_configs.idx_end_oos_preds = self.__get_global_idx_from_date__(self.data_configs.idx_end_oos_preds, mode='end')

        elif (type(data_configs.idx_start_train) is int and
              type(data_configs.idx_start_oos_preds) is int and
              type(data_configs.idx_end_oos_preds) is int):
            # 配置已经是整数索引时直接使用。
            self.data_configs = data_configs
        else:
            sys.exit('ERROR: idx_start_train and idx_start_end can be either int or date vars!')

        # Extract dataset samples covering the experiment period
        # 只保留本次实验涉及的时间范围，避免后续滑动窗口访问无关历史数据。
        self.dataset= self.dataset[self.data_configs.idx_start_train:self.data_configs.idx_end_oos_preds + 1]
        # Reindex dataset and store updated idxs in configs
        # 截取后重新从 0 编号，使后续切片都以实验片段内部索引为准。
        self.dataset[columns_keys['idx_global']] = np.arange(len(self.dataset))
        self.dataset[columns_keys['idx_step']] = np.arange(stop=len(self.dataset)) // self.data_configs.pred_horiz
        init_global_idx = self.dataset.index.tolist()[0]
        self.data_configs.idx_start_train = self.data_configs.idx_start_train - init_global_idx
        self.data_configs.idx_start_oos_preds = self.data_configs.idx_start_oos_preds - init_global_idx
        self.data_configs.idx_end_oos_preds = self.data_configs.idx_end_oos_preds - init_global_idx
        self.dataset.set_index(self.dataset[columns_keys['idx_global']], inplace=True)
        self.dataset = self.dataset.drop(columns=[columns_keys['idx_global']])

    def __build_test_samples_idxs__(self):
        # 每个测试样本覆盖一个 pred_horiz 长度的日前预测块。
        return np.arange(start=self.data_configs.idx_start_oos_preds,
                         stop=self.data_configs.idx_end_oos_preds,
                         step=self.data_configs.pred_horiz)

    def __instantiate_preproc__(self):
        # 当前仅支持 StandardScaler，特征和目标分别拟合，避免目标尺度影响输入特征。
        if self.data_configs.preprocess == 'StandardScaler':
            preproc = {
                'feat': StandardScaler(),
                'target': StandardScaler()
            }
        else:
            sys.exit('ERROR: selected preprocessing not implemented')

        return preproc

    def __build_recalib_dataset_batches__(self, df: pd.DataFrame, fit_preproc: bool):
        # extract features and target columns from the whole dataframe
        # 输入特征按命名前缀筛选：历史特征、未来已知特征、常量特征以及 DE 数据集专用特征。
        df_feat = df.filter(regex=features_keys['past'] + '|' + features_keys['futu'] + '|' + features_keys['const']
                                  + '|' + features_keys['f_l-1'] + '|' + features_keys['const_l-2'])
        df_target = df.filter(regex=features_keys['target'])

        # Fit preprocessing objects using the series steps before the pred_horiz (i.e., the recalibration test sample)
        if fit_preproc:
            # 标准化器只用测试块之前的数据拟合，避免把当前测试目标泄漏进训练过程。
            self.preproc['feat'].fit(df_feat[:-self.data_configs.pred_horiz])
            self.preproc['target'].fit(df_target[:-self.data_configs.pred_horiz])

        # Transform the series by preprocessing objects
        # 训练、验证和当前测试块都使用同一组 scaler 变换。
        np_feat_scaled = self.preproc['feat'].transform(df_feat)
        np_target_scaled = self.preproc['target'].transform(df_target)

        # Build scaled df
        df_feat_scaled = pd.DataFrame(data=np_feat_scaled,
                                      index=df.index,
                                      columns=df_feat.columns)
        df_target_scaled = pd.DataFrame(data=np_target_scaled,
                                        index=df.index,
                                        columns=df_target.columns)
        df_scaled = pd.concat([df_target_scaled, df_feat_scaled], axis=1)

        # store x columns names
        # 模型输入构造函数需要知道每一列的语义前缀，因此把列名写入 model_configs。
        self.x_columns_names = df_scaled.columns.tolist()
        self.model_configs['x_columns_names'] = self.x_columns_names

        # Create object used to generate samples following standard moving window
        # 目标列按 TARG__ 前缀识别，WindowGenerator 只会把这些列作为 y。
        target_col_name =[x for x in df_scaled.columns.tolist() if re.search(features_keys['target'], x)]
        self._win_gen = WindowGenerator(
            input_width=self.data_configs.steps_lag_win * self.data_configs.pred_horiz,
            label_width=self.data_configs.pred_horiz,
            shift=self.data_configs.pred_horiz,
            data_columns=df_scaled.columns,
            target_columns=target_col_name)

        # Convert the series into samples
        # 将连续时间序列切成不重叠的日前样本块：每次向前移动 label_width=pred_horiz。
        series_np = np.array(df_scaled.values).astype(np.float32)
        series_samples = np.stack([series_np[i:i + self._win_gen.total_window_size] for i in
                                  range(0, series_np.shape[0] - self._win_gen.total_window_size + 1, self._win_gen.label_width)])

        # Extract the last sample for test (by step-wise recalibration)
        # 当前 df 的最后一个窗口就是本轮要预测的测试样本。
        recalib_test_sample = np.copy(series_samples[-1:])
        # Put the other samples in the trainvali bag
        # 之前的窗口都进入训练/验证候选池。
        recalib_trainvali_samples = np.copy(series_samples[:-1])

        # Shuffle trainvali samples if requested
        if self.data_configs.shuffle_mode == 'train_vali':
            np.random.shuffle(recalib_trainvali_samples)

        # Build input/output samples for train_vali and test
        trainvali_samples_x, trainvali_samples_y = self._win_gen.split_window(recalib_trainvali_samples)
        x_test, y_test = self._win_gen.split_window(recalib_test_sample)

        # Separate samples devoted to train and vali
        # 训练集取前面的样本，验证集取最后 num_vali_samples 个样本，模拟靠近测试期的验证分布。
        x_train = np.copy(trainvali_samples_x[:-self.data_configs.num_vali_samples])
        y_train = np.copy(trainvali_samples_y[:-self.data_configs.num_vali_samples])
        vali_samples_x = np.copy(trainvali_samples_x[-self.data_configs.num_vali_samples:])
        vali_samples_y = np.copy(trainvali_samples_y[-self.data_configs.num_vali_samples:])

        # shuffle vali samples if required
        if self.data_configs.shuffle_mode == 'vali':
            p = np.random.permutation(len(vali_samples_y))
            vali_samples_x = vali_samples_x[p]
            vali_samples_y = vali_samples_y[p]

        # Instantiate recalibration object
        # 返回结构同时包含当前测试样本和当前滚动窗口下的训练/验证块。
        rec_samples = RecalibSamples(x_test=x_test, y_test=y_test)
        rec_samples.add_recal_block(x_train=x_train,
                                    y_train=y_train,
                                    x_vali=vali_samples_x,
                                    y_vali=vali_samples_y)

        return rec_samples

    @staticmethod
    def __build_target_quantiles__(target_alpha: List):
        """
        Build target quantiles from the list of alpha, including the median
        """
        # 与后处理模块保持一致：每个 alpha 对应一对上下界分位数，并额外保留 0.5 中位数。
        target_quantiles = [0.5]
        for alpha in target_alpha:
            target_quantiles.append(alpha/2)
            target_quantiles.append(1- alpha / 2)
        target_quantiles.sort()
        return target_quantiles

    @staticmethod
    def __build_alpha_quantiles_map__(target_alpha: List, target_quantiles: List):
        """
        Build the map between the alpha coverage levels and the related quantiles
        """
        # 用索引映射避免后面频繁按数值查找分位数列。
        alpha_q = {'med': target_quantiles.index(0.5)}
        for alpha in target_alpha:
            alpha_q[alpha] = {
                'l': target_quantiles.index(alpha/2),
                'u': target_quantiles.index(1-alpha/2),
            }
        return alpha_q

    def __transform_test_results__(self, results_df: pd.DataFrame):
        """
        Create datetime object till the end of last test date to setup the date_range properly
        """
        date_format = '%Y-%m-%d %H:%M'
        # 结束日期补到当天 23:00，确保生成完整的逐小时测试时间索引。
        end_date= self.dataset.iloc[self.data_configs.idx_end_oos_preds][columns_keys['Date']] + ' 23:00'
        end_date = datetime.strptime(end_date, date_format)
        test_block_timestamps = pd.date_range(start=self.dataset.iloc[self.data_configs.idx_start_oos_preds]
                                                    [columns_keys['Date']],
                                              end=end_date,
                                              freq='H')
        # Set datetime index to the dataframe
        results_df['Datetime'] = test_block_timestamps
        results_df.set_index(results_df['Datetime'], inplace=True)
        results_df.drop(columns=['Datetime'], inplace=True)

        # add target column
        # 把真实目标值补回结果 DataFrame，后续指标计算需要预测列和真实列在同一张表中。
        df_target = self.dataset.filter(regex=features_keys['target']).iloc[self.data_configs.idx_start_oos_preds:
                                                                            self.data_configs.idx_end_oos_preds+1]
        results_df[df_target.columns[0]] = df_target.values

        return results_df

    def get_exper_path(self):
        """
        returns the experiment path
        """
        # 路径结构与 experiments/tasks/<task>/<model_setup>/<run_id> 对齐。
        return os.path.join(os.getcwd(), 'experiments', 'tasks', self.data_configs.task_name,
                            self.data_configs.exper_setup, self.model_configs['run_id'])


    def __save_results__(self, test_results_df):
        """
        Save recalibration results
        """
        exper_save_path = os.path.join(self.get_exper_path(), 'results')
        os.makedirs(exper_save_path, exist_ok=True)
        # 保存前去掉目标列的 TARG__ 前缀，使后处理时目标列名与 PF_task 对齐。
        target_col_name = test_results_df.filter(regex=features_keys['target']).columns.tolist()[0]
        fn = target_col_name.replace(features_keys['target'], '')
        test_results_df.rename(columns={target_col_name: fn}, inplace=True)
        with open(exper_save_path + '/recalib_test_results-tuned-' + self.optuna_m + '.p', 'wb') as f:
            pickle.dump(test_results_df, f)


    def run_hyperparams_tuning(self, optuna_m:str='tpe', n_trials: int=1000):
        """
        Model hyperparameters tuning routine
        """
        def objective(trial):
            # Clear clutter from previous session graphs.
            tf.keras.backend.clear_session()
            # Update model configs with hyperparams trial
            # 每个 trial 都会把候选超参数写入 model_configs，然后训练一个模型并返回验证损失。
            self.model_configs = self.model_class.get_hyperparams_trial(trial=trial, settings=self.model_configs)

            # Build model using the current configs
            model = regression_model(settings=self.model_configs,
                                     sample_x=train_vali_block.x_vali[0:1])

            # Train model
            model.fit(train_x=train_vali_block.x_train, train_y=train_vali_block.y_train,
                      val_x=train_vali_block.x_vali, val_y=train_vali_block.y_vali,
                      pruning_call=TFKerasPruningCallback(trial, "val_loss"),
                      plot_history=False)

            # Compute val loss
            results = model.evaluate(x=train_vali_block.x_vali, y=train_vali_block.y_vali)
            return results

        # start from first train sample
        init_sample = 0
        # employ validation set till first test sample
        # 调参只使用第一个测试样本之前的数据，得到一组固定超参数供整个滚动实验复用。
        test_sample_idx = self.test_set_idxs[0]
        train_vali_block = self.__build_recalib_dataset_batches__(
            self.dataset[init_sample:test_sample_idx + self.data_configs.pred_horiz],
            fit_preproc=True).recalibBlocks[0]

        if optuna_m == 'grid_search':
            # 当前实现只支持网格搜索；搜索空间由具体模型类给出。
            search_space = self.model_class.get_hyperparams_searchspace()
            sampler = optuna.samplers.GridSampler(search_space)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=20)
        else:
            sys.exit('unknown hyperparam search mode')

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        # Unique identifier of the study.
        study_name = (self.data_configs.task_name
                      + self.model_configs['model_class'] + '-'
                      + self.model_configs['PF_method']
                      + '-' + optuna_m)
        storage_name="sqlite:///db.sqlite3"

        # Optuna study 会复用同名 sqlite 记录，便于中断后继续调参。
        study = optuna.create_study(direction="minimize",
                                    sampler=sampler,
                                    pruner=pruner,
                                    storage= storage_name,  # Specify the storage URL here.
                                    study_name=study_name,
                                    load_if_exists=True
                                    )

        timeout = 3600 * 24.0 * 7  # 7 days
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")

        print("Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            # store best hyper in the config dict
            # 把最佳 trial 的超参数写回配置，最后转成可保存的超参数字典。
            self.model_configs[key] = value

        return self.model_class.get_hyperparams_dict_from_configs(self.model_configs)

    def get_model_hyperparams(self, method, optuna_m='tpe'):
        # method 控制是重新调参还是加载已保存的 tuned_hyperp-*.json。
        self.optuna_m = optuna_m
        self.hyper_mode = method
        path = os.path.join(self.get_exper_path(), 'tuned_hyperp-' + optuna_m + '.json')
        if method=='optuna_tuner':
            print('-----------------------------------------')
            print('Starting optuna tuner')
            model_hyperparams= self.run_hyperparams_tuning(optuna_m=optuna_m)
            print('-----------------------------------------')
            # save model hyperparams to json
            # 调参完成后把最佳超参数保存到当前实验目录，后续 load_tuned 可直接复用。
            with open(path, 'w') as f:
                json.dump(model_hyperparams, f)

            return model_hyperparams

        elif method=='load_tuned':
            print('-----------------------------------------')
            print('Loading tuned hyperparams')
            print('-----------------------------------------')
            # 复现实验默认走这里，避免重新进行昂贵的 Optuna 搜索。
            with open(path) as f:
                return json.load(f)
        else:
            sys.exit('ERROR: uknown hyperparam method')

    def run_recalibration(self, hyper_mode:str, plot_history=False):
        """
        Main recalibration loop
        """
        # Get model hyperparameters (previously saved or by tuning)
        # 每个 run_id 只加载/搜索一次超参数；后面每个测试样本都会重新训练模型权重。
        model_hyperparams=self.get_model_hyperparams(method=hyper_mode, optuna_m=self.model_configs['optuna_m'])

        print('------------------------------------------------------------------------------')
        print('Starting recalibration of config: ' + str(self.model_configs['PF_method']))
        print('------------------------------------------------------------------------------')

        # List to store results over recalibration
        ensem_test_PIs=[]

        # Iterate over test samples
        for i_t in range(self.test_set_idxs.shape[0]):
            # 清理上一轮 TensorFlow 图，减少长时间滚动训练时的显存/内存累积。
            tf.keras.backend.clear_session()
            print('Recalibrating test sample: ' + str(i_t+1) + '/' + str(self.test_set_idxs.shape[0]))
            test_sample_idx = self.test_set_idxs[i_t]
            # Set index of first train sample, depending on the config
            # keep_past_train_samples=True 表示扩张窗口训练；False 表示滚动窗口训练。
            init_sample = 0 if self.data_configs.keep_past_train_samples else i_t * self.data_configs.pred_horiz

            # Build the current recalibratin batch including preprocessing (preprocess option)
            # 当前切片从 init_sample 到本轮测试块结束，最后一个窗口作为测试，其前面作为训练/验证。
            rec_samples = self.__build_recalib_dataset_batches__(
                self.dataset[init_sample:test_sample_idx+self.data_configs.pred_horiz],
                                         fit_preproc=True)

            # Get first rec_block in list
            rec_block = rec_samples.recalibBlocks[0]

            # Merge model configs and hyperparams tuning into the settings dict
            # settings 是模型、训练、分位数、ensemble 全部配置的统一入口。
            settings = {**self.model_configs, **model_hyperparams}
            # Create ensemble handler
            ensemble = Ensemble(settings=settings)

            # List to store ensemble components preds
            preds_test_e = []

            # Create and fit the ensemble components
            for e in range(settings['num_ense']):
                # 同一个测试样本下训练多个 ensemble 组件，再聚合它们的输出。
                tf.keras.backend.clear_session()
                model = regression_model(settings=settings,
                                         sample_x=rec_samples.x_test)

                model.fit(train_x=rec_block.x_train, train_y=rec_block.y_train,
                          val_x=rec_block.x_vali, val_y=rec_block.y_vali,
                          plot_history=plot_history
                          )

                # Store ensemble component prediction on test sample
                preds_test_e.append(model.predict(rec_samples.x_test))

            # Aggregate ensemble predictions
            # 对点预测/QR/分布式输出采用不同聚合方式，由 Ensemble 内部按 PF_method 分派。
            ensem_preds_test = ensemble.aggregate_preds(preds_test_e)

            # Build and store the prediction quantiles for the current test samples using the selected method
            # 所有模型最终都会被转换成统一的 [24小时, 目标分位数] 输出。
            ens_p = ensemble.get_preds_test_quantiles(preds_test=ensem_preds_test)
            rescaled_PIs = {}
            for i in range(ens_p.shape[-1]):
                # 模型在标准化目标尺度上训练，保存前要反变换回真实电价尺度。
                rescaled_PIs[self.model_configs['target_quantiles'][i]] = self.preproc['target'].inverse_transform(
                    ens_p[:, i:i + 1])[:, 0]
            results_df = pd.DataFrame(rescaled_PIs)
            ensem_test_PIs.append(results_df)

        test_results_df = self.__transform_test_results__(pd.concat(ensem_test_PIs, axis=0))

        # Save results to file
        # 最终文件供 exec_qra_cp.py 读取并继续做 QRA/CP 后处理。
        self.__save_results__(test_results_df)

        # Send email
        send_email=False #before activating this function, set sender/recipient in the declaration (email_box.py)
        if send_email:
            exper_id = self.data_configs.task_name + '--' + settings['PF_method']
            send_experimentcompleted_email(exper_id=exper_id)

        # Return test predictions
        return test_results_df
