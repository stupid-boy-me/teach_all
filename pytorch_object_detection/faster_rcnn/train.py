"""
默认训练入口 = 联合训练。

用法:
  python train.py --config configs/default.yaml
  python train.py --epochs 12

四步交替（论文复现 / 学习用）:
  python train_alternating.py
"""
from train_joint import main

if __name__ == "__main__":
    main()
