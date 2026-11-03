# Muon on CIFAR-10

课程大作业。在 CIFAR ResNet-18 上实现 Muon，并与 AdamW、带动量 SGD 对比。二维权重走 Muon，分类头、偏置和 BatchNorm 走 AdamW。

不在范围内的事：复现 Moonlight / Kimi 的大模型预训练。原方法在 NanoGPT 和 CIFAR 上的证据已经够用。

## 算法

Muon（Keller Jordan）对隐藏层矩阵做两件事：

1. 动量，系数 0.95，带 Nesterov。
2. 用五次 Newton–Schulz 迭代近似极分解。系数 `(3.4445, -4.7750, 2.0315)` 故意把零点斜率做得很陡，奇异值不会精确收到 1，所以正交残差不会变成 0。卷积先摊成 `(out, in·k·k)` 再正交化。

Moonlight（arXiv:2502.16982）补的两处固定打开，不拿来当创新点：

- 解耦权重衰减。
- 更新 RMS 对齐：正交化之后乘 `0.2 * sqrt(max(行, 列))`，使更新 RMS 接近 AdamW，从而可以共用学习率。

创新点只动 Newton–Schulz 步数（默认 5，消融用 1、2、3、5）。随机矩阵上，步数增加到 2 以后残差会来回晃，不会单调下降。这是系数本身的性质，报告里按测量结果写。

## 环境

使用已有的 `jkh` 环境（PyTorch 2.4，CUDA）。不依赖 torchvision。

```bash
conda activate jkh
cd muon-cifar
python test_muon.py
```

空闲 GPU 用 `CUDA_VISIBLE_DEVICES` 指定。当前机器上 0 和 1 是空的。

## 训练

默认学习率：SGD `0.1`，AdamW 和 Muon `1e-3`。动量：SGD `0.9`，Muon `0.95`。权重衰减都是 `5e-4`。5 个 epoch 线性 warmup，之后 cosine 降到 0。这些是起点，正式表要用短扫描选定学习率，再跑完整预算。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --optimizer muon --epochs 100 --seed 0 --out runs/muon_s0
CUDA_VISIBLE_DEVICES=0 python train.py --optimizer adamw --epochs 100 --seed 0 --out runs/adamw_s0
CUDA_VISIBLE_DEVICES=0 python train.py --optimizer sgd --epochs 100 --seed 0 --out runs/sgd_s0
```

每个 `runs/<name>/` 里有 `config.json`、`history.jsonl` 和验证集最好的 `best.pt`。

Newton–Schulz 步数消融：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --optimizer muon --ns-steps 2 --epochs 100 --seed 0 --out runs/muon_ns2_s0
```

## 实验顺序

1. 每种优化器扫三个学习率，各跑 15 个 epoch、1 个种子，留下测试准确率最高的学习率。
2. 用选定的学习率跑 100 个 epoch、3 个种子。SGD 扫 `{0.05, 0.1, 0.2}`，AdamW 和 Muon 扫 `{3e-4, 1e-3, 3e-3}`。
3. Muon 固定第 2 步的学习率，只改 `--ns-steps` 为 1、2、3、5。

## 提案

`proposal/提案.md` 是第 10 周前要交的一页。目标、做法、那一个旋钮、共同完成的分工都写了。成员姓名空着，交之前填上。
