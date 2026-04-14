# -*- coding: utf-8 -*-
"""检查当前 PyTorch 与 CUDA 环境。"""

import torch


def main() -> None:
    """打印当前机器上的 PyTorch 与 CUDA 基本信息。"""
    # 输出 PyTorch 与 CUDA 的基础状态，便于快速核对训练环境。
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"GPU 数量: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        # 当存在 GPU 时输出首张显卡名称。
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU 型号: None")


if __name__ == "__main__":
    main()
