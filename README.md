# enhanced-transformer-power

这是你的电力价格预测 Transformer 增强研究仓库，当前以论文
[A Transformer approach for Electricity Price Forecasting](https://arxiv.org/abs/2403.16108)
公开实现为基线，并在此基础上继续做版本化研发。

## 仓库目标

- 保留原始基线代码作为可对照起点
- 在不破坏现有训练入口的前提下逐步工程化
- 为后续模型增强、实验记录和结果管理提供稳定结构

## 当前目录结构

```text
.
├─ docs/                 # 项目文档与结构说明
├─ scripts/              # 辅助脚本
├─ src/                  # 核心源码
├─ tests/                # 测试目录
├─ data/                 # 数据集目录（已忽略）
├─ models/               # 模型中间产物（已忽略）
├─ best_models/          # 最优模型产物（已忽略）
├─ runs/                 # 训练日志（已忽略）
└─ results/              # 结果导出目录（已忽略）
```

详细说明见 [docs/PROJECT_STRUCTURE.md](/D:/WorkSpace/KeYan/epf-transformers-main/docs/PROJECT_STRUCTURE.md)。

## 快速开始

安装基础依赖：

```bash
pip install -r requirements.txt
```

如需按 `pyproject.toml` 安装：

```bash
pip install -e .[dev]
```

训练模型：

```bash
python -m src.train
```

运行基准评估：

```bash
python -m src.benchmark
```

检查 PyTorch 与 CUDA 环境：

```bash
python scripts/check_torch_env.py
```

## 依赖说明

当前仓库保留了原始项目的最小依赖：

- `torch==1.13.1`
- `transformers==4.26.0`

原项目还依赖 `epftoolbox`。如需完整复现实验，可执行：

```bash
git clone https://github.com/jeslago/epftoolbox.git
cd epftoolbox
git checkout 7456ab84b42240b9c2519fb3b1bbbc52868a0817
pip install .
```

## 后续建议

- 将训练超参数迁移到独立配置文件
- 修复并统一 `src/` 内历史文件的编码与导入方式
- 增加最小可运行测试，保证每次改动可回归验证

## 引用

如果你使用了原始基线思路，请引用原论文：

```bibtex
@misc{gonzalez2024transformer,
      title={A Transformer approach for Electricity Price Forecasting},
      author={Oscar Llorente Gonzalez and Jose Portela},
      year={2024},
      eprint={2403.16108},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
