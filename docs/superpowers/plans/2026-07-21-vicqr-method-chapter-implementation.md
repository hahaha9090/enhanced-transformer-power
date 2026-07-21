# VICQR方法章节落地 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将作者已确认的第三章设计同步到论文记忆体系，并建立体现VICQR完整方法对象的当前中文稿骨架。

**Architecture:** 以`.paper/blueprint.md`继续作为唯一研究事实来源，只在已锁定的第三章结构内增加方法表达目标；`.paper/writing_protocol.md`负责把该目标转成写作和审查规则。当前中文稿只建立五章结构及第二、第三章段落职责，不补造尚未从实现、协议或结果文件核验的数学细节。

**Tech Stack:** Markdown、YAML、PowerShell文本检查、Git版本记录

## Global Constraints

- 第三章的首要任务，是使审稿人将VICQR识别为一个定义完整、逻辑闭合、科学可信的新方法，而不是若干经验规则的拼接。
- 可复现性作为方法完整性的检验条件保留，但不作为第三章的叙事中心。
- 论文正文、附录和补充材料均不使用伪代码或程序代码式Algorithm框。
- 方法图必须采用与真实模型一致的多层、多分支结构，并包含双证据路径、风险门控、不动作分支和在线反馈闭环。
- 不改变已锁定的中心问题、实验数字、证据批次、五章结构和结论强度。
- 不得把VICQR写成“OCQR先输出区间、VICQR再附加处理”的插件式串联模型。
- 只暂存和提交本计划明确列出的文件，不覆盖工作区中的其他用户修改。

---

### Task 1: 对齐蓝图与写作协议中的方法章节目标

**Files:**
- Modify: `.paper/blueprint.md`
- Modify: `.paper/writing_protocol.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-21-vicqr-method-chapter-design.md`中的已确认设计；蓝图v1.3中的模型定义、证据边界和第三章结构。
- Produces: 蓝图v1.4与写作协议v1.1，供当前稿、状态文件和后续`nature-writing`任务统一引用。

- [ ] **Step 1: 在蓝图中锁定第三章的首要表达目标**

  将蓝图版本由`v1.3`更新为`v1.4`。在第11节第三章说明中加入以下已确认内容，不改变3.1—3.4标题与职责：

  > 第三章的首要任务，是使审稿人将VICQR识别为一个具有明确输入、状态、候选动作、证据评价、决策机制、风险控制、输出和在线更新闭环的完整新方法，而不是若干经验规则的拼接。可复现性是方法完整性的检验条件，但不是本章的叙事中心；本章通过统一方法对象、必要数学定义和机制关系建立科学可信度。

  在第13.3节补充：各机制必须先被放入统一决策闭环，再分别解释其数学定义和作用；不得把“模块数量多”当作创新性证据。

- [ ] **Step 2: 将同一目标转化为写作执行规则**

  将协议版本由`v1.0`更新为`v1.1`，上位依据改为蓝图`v1.4`。在第7.4节明确第三章的推荐展开顺序：

  1. 统一定义VICQR的输入、输出和预测时点信息集；
  2. 定义波动状态与有符号分位数动作空间；
  3. 定义留一市场证据与因果可用本地反馈的评价关系；
  4. 定义风险门控、动作接受或拒绝、不动作分支、最终区间和观测到达后的更新；
  5. 将数学定义、参数和信息时序作为方法完整性检查，而不把代码级复现写成核心贡献。

  在第12节把“可复现性”质量门重述为“方法完整性与可复核性”，检查对象保持为公式、信息集、决策规则、参数和方法图。

- [ ] **Step 3: 验证没有改变论文事实或章节结构**

  Run:

  ```powershell
  rg -n "蓝图版本|协议版本|第三章的首要任务|可复现性|3\.1 Problem Formulation|3\.4 Risk-Gated" .paper/blueprint.md .paper/writing_protocol.md
  git diff --check -- .paper/blueprint.md .paper/writing_protocol.md
  ```

  Expected: 蓝图显示`v1.4`，协议显示`v1.1`并引用`v1.4`；3.1—3.4标题不变；无空白错误。

- [ ] **Step 4: 只提交本任务文件**

  ```powershell
  git add -- .paper/blueprint.md .paper/writing_protocol.md
  git commit -m "明确VICQR方法章节的完整模型表达目标"
  ```

### Task 2: 建立当前中文稿的五章与方法论证骨架

**Files:**
- Create: `paper_outputs/word/vicqr_current/vicqr_manuscript_cn.md`

**Interfaces:**
- Consumes: 蓝图v1.4的五章结构、C9—C18的证据边界、Fig1—Fig3的方法与诊断图职责。
- Produces: 当前唯一中文稿，供后续`nature-writing`逐段起草；正文事实仍由蓝图和主张库控制。

- [ ] **Step 1: 建立固定的一级、二级标题**

  文件必须按以下顺序建立：`Abstract`、`1. Introduction`、第二章及2.1—2.3、第三章及3.1—3.4、第四章及4.1—4.5、`5. Conclusion`。标题逐字采用蓝图v1.4，不新增独立相关工作、讨论或伪代码小节。

- [ ] **Step 2: 为第二章写入段落级论证职责**

  - 2.1依次安排：预测任务与信息集；因果可用高波动状态；名义覆盖与条件诊断指标；“问题存在”的判定规则。
  - 2.2依次安排：OCQR有效能力边界；最新聚合点估计中的高波动欠覆盖；旧完整批次尾部与市场异质性辅助观察；不能推广到全部市场—小时的边界。
  - 2.3依次安排：从欠覆盖推出状态识别要求；从尾部不平衡推出有符号动作要求；从稀疏状态推出双证据要求；从局部恶化风险推出风险门控和不动作要求。

  每个段落条目均注明其可使用的主张ID和图表ID，不在第二章写入VICQR效果或机制消融结果。

- [ ] **Step 3: 为第三章写入统一模型的段落级论证职责**

  - 章节开篇先给出VICQR的统一定义：输入与信息集→波动状态→候选有符号动作→双证据评价→风险门控→区间输出→观测后更新。
  - 3.1定义任务、统一模型对象、闭环关系，以及VICQR如何吸收经研究确认有效的在线边际校准原理并重构决策过程。
  - 3.2定义状态、上下尾风险、动作变量、收缩/扩张/尾部转移/不动作的同一动作空间和proper-score评价目标。
  - 3.3定义留一市场证据、本地历史反馈、因果截止与融合评价，明确两条证据路径共同服务于候选动作决策。
  - 3.4定义风险上界、动作接受或拒绝、不动作分支、区间结构约束、最终输出、观测到达后的更新和复杂度说明。

  对尚未从实现或协议核验的公式、参数和阈值，使用“定义需求”条目标明需要核验的对象及来源，不填写臆测数值或公式。

- [ ] **Step 4: 验证骨架体现完整模型而非规则清单**

  Run:

  ```powershell
  rg -n "输入|信息集|波动状态|候选动作|跨市场证据|本地反馈|风险门控|不动作|区间输出|在线更新" paper_outputs/word/vicqr_current/vicqr_manuscript_cn.md
  rg -n "伪代码|Algorithm 1|OCQR先输出|附加处理|轻量插件" paper_outputs/word/vicqr_current/vicqr_manuscript_cn.md
  git diff --check -- paper_outputs/word/vicqr_current/vicqr_manuscript_cn.md
  ```

  Expected: 第一条检查覆盖九类完整方法对象；第二条无匹配；无空白错误。

- [ ] **Step 5: 只提交当前中文稿**

  ```powershell
  git add -- paper_outputs/word/vicqr_current/vicqr_manuscript_cn.md
  git commit -m "建立VICQR中文论文与方法论证骨架"
  ```

### Task 3: 同步当前状态、稿件入口和修订记录

**Files:**
- Modify: `.paper/status.md`
- Modify: `.paper/claims.yml`（仅同步`source_blueprint`，不改主张内容或状态）
- Modify: `.paper/figures.yml`（仅同步`source_blueprint`，不改图表职责或证据边界）
- Modify: `.paper/revision_history.yml`
- Modify: `paper_outputs/word/vicqr_current/CURRENT_MANUSCRIPT.md`

**Interfaces:**
- Consumes: 蓝图v1.4、写作协议v1.1和已创建的`vicqr_manuscript_cn.md`。
- Produces: 一致的当前阶段导航、现行稿入口和R15修订记录。

- [ ] **Step 1: 更新状态文件**

  在“已经锁定的内容”中增加第三章首要目标；在“已经完成的工作”中记录方法章节设计对齐和当前中文稿骨架建立；将当前下一步改为依据定义需求核验实现、协议和结果文件，再逐段起草第二、第三章。

- [ ] **Step 2: 更新当前稿入口**

  将状态改为“当前中文稿骨架已建立”，登记：

  - 正文文件：`vicqr_manuscript_cn.md`
  - 上位蓝图：`.paper/blueprint.md v1.4`
  - 写作协议：`.paper/writing_protocol.md v1.1`
  - 当前可写范围：五章结构及第二、第三章段落级骨架；量化结论仍受C9—C23状态约束。

- [ ] **Step 3: 新增R15修订记录**

  将`claims.yml`和`figures.yml`的`source_blueprint`从v1.3同步为v1.4，不改变任何主张状态、数字、图表职责或证据边界。R15只记录本轮经作者确认的变化：第三章首要目标由“强调可复现”校正为“建立完整、可信、创新的方法对象”；可复现性保留为完整性检查；建立当前中文稿骨架；未改变中心问题、数字、证据批次和五章结构。

- [ ] **Step 4: 验证跨文件版本与入口一致**

  Run:

  ```powershell
  rg -n "v1\.3|v1\.4|v1\.0|v1\.1|vicqr_manuscript_cn\.md|R15" .paper/status.md .paper/blueprint.md .paper/writing_protocol.md .paper/claims.yml .paper/figures.yml .paper/revision_history.yml paper_outputs/word/vicqr_current/CURRENT_MANUSCRIPT.md
  git diff --check -- .paper/status.md .paper/revision_history.yml paper_outputs/word/vicqr_current/CURRENT_MANUSCRIPT.md
  ```

  Expected: 现行入口、`claims.yml`和`figures.yml`引用蓝图v1.4，现行入口引用协议v1.1；历史修订记录中的旧版本号保留；R15存在；无空白错误。

- [ ] **Step 5: 只提交导航与记录文件**

  ```powershell
  git add -- .paper/status.md .paper/claims.yml .paper/figures.yml .paper/revision_history.yml paper_outputs/word/vicqr_current/CURRENT_MANUSCRIPT.md
  git commit -m "同步VICQR当前稿状态与修订记录"
  ```

### Task 4: 完成全局一致性和越界检查

**Files:**
- Verify: `.paper/blueprint.md`
- Verify: `.paper/writing_protocol.md`
- Verify: `.paper/status.md`
- Verify: `.paper/references.yml`
- Verify: `.paper/claims.yml`
- Verify: `.paper/figures.yml`
- Verify: `.paper/revision_history.yml`
- Verify: `paper_outputs/word/vicqr_current/CURRENT_MANUSCRIPT.md`
- Verify: `paper_outputs/word/vicqr_current/vicqr_manuscript_cn.md`

**Interfaces:**
- Consumes: Tasks 1—3的全部输出。
- Produces: 可继续逐段写作、且没有证据越界和叙事回退的统一入口。

- [ ] **Step 1: 扫描禁止表述与过程残留**

  Run:

  ```powershell
  rg -n "VICQR 是 OCQR 之后的简单后处理层|OCQR 加上 VICQR|只是为 OCQR 增加一个分位数|全面解决高波动欠覆盖|普遍条件覆盖保证|统计显著|5\.90%|6\.88%|Algorithm 1|伪代码" .paper paper_outputs/word/vicqr_current
  ```

  Expected: 仅允许在蓝图、主张库或协议的“禁止使用/当前不支持”审计语境中出现；当前中文稿不得出现这些表达。

- [ ] **Step 2: 检查当前稿的章节数量与职责**

  Run:

  ```powershell
  rg -n "^# (Abstract|[1-5]\.)|^## [234]\.[1-5]" paper_outputs/word/vicqr_current/vicqr_manuscript_cn.md
  ```

  Expected: 五个编号一级章节、摘要以及蓝图锁定的2.1—2.3、3.1—3.4、4.1—4.5全部存在，顺序一致。

- [ ] **Step 3: 检查最终改动范围**

  Run:

  ```powershell
  git status --short
  git log -3 --oneline
  ```

  Expected: 本计划只产生明确列出的论文记忆与当前稿文件变更；既有用户修改保持原样；最近提交信息均为中文。
