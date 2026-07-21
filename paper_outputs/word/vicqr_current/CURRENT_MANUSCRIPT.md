# VICQR 当前论文稿唯一入口

- 状态：当前中文稿骨架已建立，第二、第三章进入定义核验与逐段起草阶段。
- 更新日期：2026-07-21
- 上位蓝图：`.paper/blueprint.md` v1.4
- 写作协议：`.paper/writing_protocol.md` v1.1
- 当前证据批次：七市场最新恢复聚合点估计；旧本地完整批次仅作辅助机制证据。
- 当前正文文件：`vicqr_manuscript_cn.md`
- 当前可写范围：五章结构及第二、第三章段落级骨架；量化结论仍受 `.paper/claims.yml` 中 C9—C23 的状态和证据边界约束。

## 当前稿规则

1. VICQR 当前论文稿只在本目录创建、续写和版本化。
2. `paper_outputs/word/` 下本目录之外的 Word、Markdown 和论文框架均为历史材料，未经作者明确指定不得作为当前稿继续续写。
3. 当前正文必须读取 `.paper/status.md`、`.paper/blueprint.md`、`.paper/writing_protocol.md`、`.paper/references.yml`、`.paper/claims.yml`、`.paper/figures.yml` 和 `.paper/revision_history.yml`。
4. 正文采用蓝图 v1.4 锁定的五章结构，不使用伪代码或 Algorithm 框；方法使用多层、多分支、带在线反馈闭环的复杂架构图表达。

## 锁定的五章结构

1. `Introduction`
2. `Problem Formulation and Diagnosis of Interval Reliability under High-Volatility Conditions`
3. `VICQR: A Volatility-Informed Conditional Quantile Reallocation Model for Online Interval Calibration`
4. `Experimental Evaluation and Discussion`
5. `Conclusion`

第二章只证明问题并推出设计要求；第三章只定义完整VICQR模型；第四章集中报告主要效果、跨市场一致性、机制消融、替代解释、风险门控、外部证据和适用边界。

## 当前可用结论

- OCQR 在两个主要区间水平上存在平均高波动欠覆盖的恢复聚合点估计。
- VICQR 的恢复聚合点估计表现为高波动覆盖恢复、Winkler 分数降低、区间收窄和两个水平下 7/7 市场方向一致。
- 当前不能声称统计显著、普遍条件覆盖保证、全面优于近期强基线，或在当前批次保持总体覆盖。
- 旧完整批次中的状态、证据融合和风险门控控制只作为辅助机制证据，最终定量表述仍需与主结果批次对齐。

## 下一步

按照 `vicqr_manuscript_cn.md` 中的定义需求核验实现、协议和审计文件，锁定第二、第三章所需的符号、参数、信息时序和决策规则；核验后优先扩写问题诊断与 VICQR 方法，再写基于恢复聚合点估计的结果与讨论。
