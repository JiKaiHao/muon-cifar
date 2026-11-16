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

## 实验结果（已跑完，2026-11-15，刘东旭）

上面写的三阶段实验已经全部跑完，一共 30 个 run，0 失败。原始数据在 `runs/`，汇总在 `runs/summary.md`，两张图在 `runs/main/test_acc.png` 和 `runs/ablation/steps_summary.png`。

### 环境说明（这台机器上 `jkh` 环境不存在）

实际用的是 PyTorch 2.4.1 + CUDA 12.1（Python 3.10）。CIFAR-10 用 `scripts/fetch_cifar.py` 下到 `data/`。原来的 `sweep_lr.py` / `run_main.py` / `run_ablation.py` 都写死只用 2 张卡，新增的 `scripts/run_all_8gpu.py` 把同样的协议排到 8 张卡上——epochs、warmup、学习率、`ns_steps`、批量 128、workers 4 全部照抄原脚本，只改了显卡分配，另外加了第四个阶段跑 SGD 的第二个学习率。

```bash
python scripts/run_all_8gpu.py                       # 全部跑完，断点续跑
python scripts/summarize.py                          # 生成 runs/summary.md
python scripts/plot_ablation.py                      # 步数消融的两张图
```

### 一、学习率扫描（15 epoch，seed 0）

| 优化器 | 学习率 | best test |
| --- | --- | --- |
| SGD | 0.05 | 91.08 |
| SGD | 0.1 | **91.38** |
| SGD | 0.2 | 89.89 |
| AdamW | 3e-4 | 90.61 |
| AdamW | 1e-3 | **90.83** |
| AdamW | 3e-3 | 90.68 |
| Muon | 3e-4 | 92.28 |
| Muon | 1e-3 | 93.27 |
| Muon | 3e-3 | **93.44** |

AdamW 和 Muon 选出的学习率和之前一致（1e-3、3e-3）。**SGD 这里和现在 `paper/experiment.html` 里写的不一样**：正文写的是"SGD 选 0.05，因为 91.06% 略高于 0.1 的 91.05%"，这次重跑是 0.1 (91.38) 反超 0.05 (91.08)。差 0.3 个百分点，所以下面 100 epoch 时把两个都跑了。

### 二、正式对比（100 epoch，seed 0/1/2，批量 128）

| 优化器 | 学习率 | best test | last test | 每 epoch | 墙钟 |
| --- | --- | --- | --- | --- | --- |
| SGD | 0.05 | 94.88 ± 0.10 | 94.82 ± 0.09 | 7.4 s | 12.3 min |
| AdamW | 1e-3 | 93.93 ± 0.16 | 93.86 ± 0.19 | 7.5 s | 12.6 min |
| Muon | 3e-3 | 94.55 ± 0.14 | 94.49 ± 0.14 | 21.4 s | 35.6 min |

SGD 第二个学习率也跑满了 100 epoch：

| 学习率 | best test | last test | 墙钟 |
| --- | --- | --- | --- |
| 0.05 | 94.88 ± 0.10 | 94.82 ± 0.09 | 12.3 min |
| 0.1 | **95.10 ± 0.14** | 95.07 ± 0.15 | 15.1 min |

三件事要注意：

1. **Muon 没有赢过 SGD**。94.55 低于 0.05 学习率的 94.88，更低于 0.1 的 95.10，墙钟还慢 2.9 倍。Muon 只明显好于 AdamW（+0.62）。
2. SGD 两个学习率的差距在 100 epoch 下依然存在（0.22），三个种子方向一致，不是噪声。所以选 0.05 那个结论建议改。
3. `best test` 是 100 个 epoch 里的最高值，`last test` 是最后一轮的准确率，两者只差 0.06 左右，说明不是靠挑 epoch 挑出来的。

### 三、Newton–Schulz 步数消融（100 epoch，学习率 3e-3，seed 0/1/2）

| ns_steps | best test | 末轮正交残差 | 末轮更新 RMS | 每 epoch |
| --- | --- | --- | --- | --- |
| 1 | 93.99 ± 0.06 | 0.9455 | 0.0505 | 17.8 s |
| 2 | 94.37 ± 0.04 | 0.8074 | 0.1028 | 18.7 s |
| 3 | **94.86 ± 0.19** | 0.5934 | 0.1491 | 19.4 s |
| 5 | 94.55 ± 0.14 | 0.3657 | 0.1810 | 21.4 s |

（第 5 步直接复用 `runs/main/muon_s*`，没有重复跑。）

这是全套实验里最有意思的一条：**正交残差随步数单调下降（0.95 → 0.37），但测试准确率在第 3 步见顶，第 5 步反而掉 0.31**。也就是说，把更新做得更接近极分解，并没有换来更好的准确率，还多花 10% 时间。这正好对上 `paper/experiment.html` 里引的 Liu 等的说法——他们观察到 10 步比 5 步更正交但准确率没有更好。我们的数据在 3 → 5 步这一段复现了同样的现象。

`update_rms` 随步数上升（0.05 → 0.18）是预期内的：牛顿–舒尔茨把奇异值拉向 1，更新矩阵的典型元素幅度自然变大，这也是为什么 RMS 对齐系数要固定打开。

### 需要一起定的两件事

1. **SGD 学习率 0.05 还是 0.1**。数据倾向 0.1，但正文已经写了 0.05，改不改要一起定。上面的表两种都有，改一个数字即可。
2. **怎么叙述 Muon 没有超过 SGD**。如实写是一个干净的可复现结果，而且和"步数不是越多越好"这条线正好呼应；但不建议为了写成 Muon 更好去调 SGD 的学习率或权重衰减。

