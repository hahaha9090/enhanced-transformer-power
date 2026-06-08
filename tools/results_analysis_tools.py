"""
Utility functions for results analysis
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import tools.stat_tests_tools as stt
import tools.plot_tools as pl
from tools.prediction_quantiles_tools import (build_alpha_quantiles_map,
                                              build_target_quantiles)


models_names_short = {'CP-DNN': 'CP',
                     # 统一缩写模型名，便于画图和表格列名更紧凑。
                     'QRA-DNN': 'QRA',
                     'N-DNN': 'Norm',
                     'JSU-DNN': 'Jsu',
                     'STU-DNN': 'Stu',
                     'QR-DNN': 'QR',
                     'CQ-N-DNN': 'CQN',
                     'CQ-JSU-DNN':'CQJ',
                     'CQ-STU-DNN': 'CQS',
                     'CQ-QR-DNN': 'CQR',
                     'OCQ-N-DNN': 'OCQN',
                      'OCQ-JSU-DNN': 'OCQJ',
                      'OCQ-STU-DNN': 'OCQS',
                      'OCQ-QR-DNN': 'OCQR',
                      }


def winkler_score(y_true, pred_PI_l, pred_PI_u, alpha):
    # Winkler 分数同时惩罚区间宽度和未覆盖样本；分数越小表示区间越窄且覆盖越可靠。
    score = []
    delta = pred_PI_u - pred_PI_l
    for t in range(y_true.shape[0]):
        if y_true[t] < pred_PI_l[t]:
            score.append(delta[t] + (2/alpha)*(pred_PI_l[t] - y_true[t]))
        elif y_true[t] > pred_PI_u[t]:
            score.append(delta[t] + (2/alpha) * (y_true[t] - pred_PI_u[t]))
        else:
            score.append(delta[t])
    score = np.array(score)
    return score


def pinball_score(labels, pred_quantiles, quantiles):
    # Pinball loss 是分位数预测的标准损失；这里对所有目标分位数求平均。
    loss = []
    for i, q in enumerate(quantiles):
        error = np.subtract(labels, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        loss.append(np.expand_dims(loss_q,-1))
    loss = np.mean(np.concatenate(loss, axis=-1), axis=-1)
    return loss


class ExperAnalysis():
    """
    Compute kpis and plots
    """
    def __init__(self, PF_task:str, run_id:str,
                 kupiec_significance_level=0.05, kupiec_max_LR_UC=20
                 ):
        # load results
        # 读取 exec_qra_cp.py 保存的聚合 pickle，里面包含所有方法的结果 DataFrame 和实验元信息。
        path = os.path.join(os.getcwd(), 'experiments', 'tasks', PF_task)
        with open(path + '/' + run_id + '_aggr_results.p', 'rb') as fp:
            exper_results = pickle.load(fp)

        # Store configs
        # target_alpha 是失覆盖率列表；pred_horiz 通常为 24，表示日前电价 24 小时预测。
        self.target_name = PF_task
        self.pred_horiz = exper_results['pred_horiz']
        self.target_alpha= exper_results['target_alpha']

        # 构造分位数列表和 alpha 到上下界列索引的映射，后续所有指标都依赖这个映射。
        self.target_quantiles = build_target_quantiles(self.target_alpha)
        self.alpha_q_map = build_alpha_quantiles_map(target_alpha=self.target_alpha,
                                                     target_quantiles=self.target_quantiles)
        self.kupiec_significance_level=kupiec_significance_level
        self.kupiec_max_LR_UC = kupiec_max_LR_UC
        self.experiments=exper_results['results']
        # 把完整模型名替换为短名，图和表更易读。
        self.experiments = {models_names_short.get(k, k): v for k, v in self.experiments.items()}
        self.exper_names = list(self.experiments.keys())

        # Initialize kpis
        self.kupiec_tests = {}
        self.hourly_LR_UC = {} # likelihood_ratio
        self.passed_kupiec_tests = {}
        self.PICP = {}
        self.winkler = {}
        self.pinball = {}
        self.mae = {}

        # Compute kpis
        self.compute_kpis(self.experiments)
        self.compute_mean_kpis()

    def compute_kpis(self, experiments):
        # Initialize dicts
        # 每个 alpha 都单独保存覆盖率、Kupiec 统计量和 Winkler 分数。
        for alpha in self.target_alpha:
            self.kupiec_tests[alpha] = {}
            self.passed_kupiec_tests[alpha] = {}
            self.PICP[alpha] = {}
            self.hourly_LR_UC[alpha] = {}
            self.winkler[alpha] = {}

        # Perform results analysis for each model configuration m_k
        for m_k, m_pred in experiments.items():
            # m_pred 的列结构是：若干预测分位数列 + 真实目标列。
            preds = m_pred.loc[:, m_pred.columns != self.target_name].to_numpy()
            self.y_true = m_pred[self.target_name]
            labels = m_pred[self.target_name].to_numpy()
            # 拉回 [天/样本, 24小时, 分位数数]，便于按小时计算覆盖率和统计检验。
            preds_h = preds.reshape(-1, self.pred_horiz, preds.shape[1])
            labels_h = labels.reshape(-1, self.pred_horiz)
            for alpha in self.target_alpha:
                kup_h = []
                LR_UC = []
                PICP_h = []
                lq_idx = self.alpha_q_map[alpha]['l']
                uq_idx = self.alpha_q_map[alpha]['u']
                for hour in range(self.pred_horiz):
                    # Compute PI violation
                    # PI_hits=True 表示真实价格落在当前 alpha 对应预测区间内。
                    PI_hits = np.logical_and(preds_h[:, hour, lq_idx] <= labels_h[:, hour],
                                             labels_h[:, hour] <= preds_h[:, hour, uq_idx])
                    # Compute kupiec test for each alpha
                    kup_results = stt.kupiec_test(PI_hits=PI_hits, alpha=alpha,
                                                  significance_level=self.kupiec_significance_level)
                    kup_h.append(kup_results)
                    LR_UC.append(np.clip(kup_results['LR_UC'], a_min=None, a_max=self.kupiec_max_LR_UC))
                    # Compute PI coverage probability from hits for each alpha
                    PICP_h.append(np.round(len(PI_hits[PI_hits == 1]) / PI_hits.shape[0], decimals=4))

                # Store results
                self.kupiec_tests[alpha][m_k] = kup_h
                self.passed_kupiec_tests[alpha][m_k] = sum([p['passed'] for p in kup_h])
                self.hourly_LR_UC[alpha][m_k] = np.array(LR_UC)
                self.PICP[alpha][m_k] = PICP_h
                # Compute winkler's score for each alpha
                self.winkler[alpha][m_k] = winkler_score(y_true=labels,
                                                         pred_PI_l=preds[:, lq_idx], pred_PI_u=preds[:, uq_idx],
                                                         alpha=alpha)
            # Compute average pinball loss for all quantiles
            self.pinball[m_k] = pinball_score(labels=labels_h, pred_quantiles=preds_h,
                                              quantiles=self.target_quantiles).flatten()
            # Compute MAE
            self.mae[m_k] = np.abs(labels - preds[:, self.alpha_q_map['med']])

        # Convert PICP, pinball and winkler scores to dataframes for later usage
        # DataFrame 形式方便后续求均值、画图、做 DM 检验。
        self.pinball_df = pd.DataFrame.from_dict(self.pinball)
        self.mae_df = pd.DataFrame.from_dict(self.mae)
        self.winkler_df = {}
        for alpha in self.target_alpha:
            self.PICP[alpha] = pd.DataFrame.from_dict(self.PICP[alpha])
            self.winkler_df[alpha] = pd.DataFrame.from_dict(self.winkler[alpha])

    @staticmethod
    def alpha_rename(name, alpha):
        # 生成带 alpha 下标的行名，例如 PICP$_{0.2}$。
        return name + '$_{' + str(alpha) + '}$'

    def compute_mean_kpis(self):
        # 汇总每个模型的均值指标，作为论文主表或附表的基础。
        KPIs = {}
        for m_k in self.exper_names:
            kpis_m = {}
            kpis_m['Pinball'] = np.round(self.pinball_df[m_k].mean(), decimals=3)
            kpis_m['MAE'] = np.round(np.mean(self.mae[m_k]), decimals=3)

            for alpha in self.target_alpha:
                # Unconditional Coverage, also named PICP (PI coverage probability)
                kpis_m[self.alpha_rename('PICP', alpha)] = 100 * np.round(self.PICP[alpha][m_k].mean(), decimals=2)
                # Kipiec_test_passed
                kpis_m[self.alpha_rename('Kupiec passed', alpha)] = np.round(self.passed_kupiec_tests[alpha][m_k], decimals=0)
                # Mean Winkler_score
                kpis_m[self.alpha_rename('Winkler', alpha)] = np.round(self.winkler_df[alpha][m_k].mean(), decimals=3)
                KPIs[m_k] = kpis_m

        self.mean_KPIs = pd.DataFrame.from_dict(KPIs)
        self.__format_latex__()

    def __format_latex__(self):
        # format data to save latex table
        # 先把数值转成固定小数位字符串，保证 LaTeX/Markdown 表格展示一致。
        self.mean_KPIs.loc['Pinball'] = self.mean_KPIs.loc['Pinball'].map('{:,.3f}'.format)
        self.mean_KPIs.loc['MAE'] = self.mean_KPIs.loc['MAE'].map('{:,.3f}'.format)
        for alpha in self.target_alpha:
            self.mean_KPIs.loc[self.alpha_rename('PICP', alpha)] = (
                self.mean_KPIs.loc[self.alpha_rename('PICP', alpha)].map('{:,.2f}'.format))
            self.mean_KPIs.loc[self.alpha_rename('Kupiec passed', alpha)] = (
                self.mean_KPIs.loc[self.alpha_rename('Kupiec passed', alpha)].astype(int).map(
                '{:,d}'.format))
            self.mean_KPIs.loc[self.alpha_rename('Winkler', alpha)] = (
                self.mean_KPIs.loc[self.alpha_rename('Winkler', alpha)].map('{:,.2f}'.format))

    # Print table in latex form
    def table_mean_kpis_latex(self):
        return self.mean_KPIs.to_latex()

    # Print table in markdown form
    def table_mean_kpis_markdown(self):
        return self.mean_KPIs.to_markdown()

    def plot_kupiec(self):
        # 对每个 alpha 画一组 24 小时 Kupiec LR_UC，红线为卡方临界值。
        for alpha in self.target_alpha:
            pl.plot_kupiec(hourly_LR_UC=self.hourly_LR_UC[alpha],
                           critical_chi_square=stt.get_critical_chi_square(self.kupiec_significance_level),
                           max_LR_UC=self.kupiec_max_LR_UC, title='α='+str(alpha))

    def plot_stepwise_PICP(self, conf_to_plot):
        # 输入可以使用完整模型名，这里先转换成初始化阶段使用的短名。
        conf_to_plot =[models_names_short[k] for k in conf_to_plot]

        formatter = FormatStrFormatter("%.2f")
        fig, axes = plt.subplots(4, 1, figsize=(4.5, 17))
        i=0
        for alpha in self.target_alpha:
            fig_picp = self.PICP[alpha][conf_to_plot].plot(ax=axes[i])
            fig_picp.grid()
            # 红色虚线表示理论目标覆盖率 1-alpha，用来直观看模型是否欠覆盖/过覆盖。
            x_values = [0, 23]
            y_values = [1 - alpha, 1 - alpha]
            axes[i].plot(x_values, y_values, linestyle="--", color='red')
            axes[i].set_ylabel('Hourly PICP - α='+str(alpha))
            axes[i].legend(loc='upper right', bbox_to_anchor=(1.55, 0.95))
            axes[i].yaxis.set_major_formatter(formatter)
            i=i+1
        axes[i-1].set_xlabel('Hour')
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(right=0.6)
        plt.show()


    def plot_DM_test_pinball(self):
        # DM 检验比较不同模型的 pinball loss 是否存在统计显著差异。
        stt.plot_multivariate_DM_test(real_price=self.y_true, forecasts_losses=self.pinball_df,
                                      title='Pinball')

    def plot_DM_test_mae(self):
        # 使用 MAE 作为损失序列做同样的模型两两比较。
        stt.plot_multivariate_DM_test(real_price=self.y_true, forecasts_losses=self.mae_df,
                                      title='MAE')

    def plot_DM_test_winkler(self):
        # Winkler 分数依赖 alpha，因此每个预测区间置信水平单独做 DM 检验。
        for alpha in self.target_alpha:
            stt.plot_multivariate_DM_test(real_price=self.y_true, forecasts_losses=self.winkler_df[alpha],
                                          title='Winkler - α=' + str(alpha))

    def plot_experiment_predictions(self, exper_name, target):
        # 画指定模型的分位数预测曲线，用于肉眼检查区间宽度和真实值覆盖情况。
        pl.plot_quantiles(self.experiments[exper_name], target)



