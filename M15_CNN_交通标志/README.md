# M15 · 卷积层构造与 BatchNormalization 实战（交通标志识别）

配套文章：`M15_限速30和限速80只差一个数字.md`

用 GTSRB（德国交通标志识别基准，43 类，真实车载摄像头采集），从一个纯 numpy 手写的卷积网络出发，讲四件事：卷积层怎么构造、卷积相对全连接省下多少参数、BatchNormalization 到底换来了什么、以及模型在 43 类上具体错在哪里。

## 目录结构

```
M15_CNN_交通标志/
├── M15_限速30和限速80只差一个数字.md   # 正文
├── code/
│   ├── prepare_data.py    # 下载 GTSRB(zip) + md5 校验 + ROI 裁切 + 分层切分
│   ├── train_cnn.py       # 从零 numpy CNN：卷积(im2col)/BN/池化/全连接 + 学习率扫描 + 2x2 对照
│   ├── gradcheck.py       # 数值中心差分校验反向传播（float64）
│   ├── analyze_errors.py  # 错例按语义家族归类，产出 error_analysis.json
│   ├── make_figures.py    # 从 stats.json + error_analysis.json + plot_data.npz 重生 fig1-9
│   └── qc_article.py      # 文章 QC：字数/黑名单/破折号/数学占比/数字一致性/掘金合规
├── data/
│   ├── stats.json         # 训练与参数账本的唯一来源
│   ├── error_analysis.json# 错例家族归类的唯一来源
│   ├── plot_data.npz      # 出图所需中间量（曲线/混淆矩阵/卷积核/测试图）
│   ├── run.log            # 训练日志
│   └── gtsrb32.npz        # 预处理后的 32x32 数据（脚本生成，不入库）
├── figures/               # fig1-9
└── requirements.txt
```

## 九张配图

| 图 | 文件 | 讲一件事 |
|---|---|---|
| 图 1 | fig1_arch.png | 网络结构全景与每层参数 |
| 图 2 | fig2_data.png | 数据来源：16 张真实样本 + 43 类不平衡分布 |
| 图 3 | fig3_conv.png | 一个 3×3×3 核在三个通道上怎么算 |
| 图 4 | fig4_ledger.png | 参数账本：448 vs 50348032；全网络参数构成 |
| 图 5 | fig5_bn.png | BatchNorm 前向：训练用批统计量，推理用滑动平均 |
| 图 6 | fig6_lrscan.png | 学习率扫描：5 档 lr × 带/不带 BN |
| 图 7 | fig7_curves.png | 全量数据 2×2 对照的损失与准确率曲线 |
| 图 8 | fig8_confusion.png | 43×43 混淆矩阵（红格标出判错的 26 张）|
| 图 9 | fig9_topconf.png | 误判结构：错在哪一族、D 组塌向哪里、限速族 8 块牌子 |

## 复现

```bash
pip install -r requirements.txt
python code/prepare_data.py     # 下载 179 MB 并预处理，产出 gtsrb32.npz
python code/gradcheck.py        # 反向传播校验，各层相对误差 2.2e-6 ~ 5.8e-6
python code/train_cnn.py        # 学习率扫描 + 2x2 全量对照，产出 stats.json + plot_data.npz
python code/analyze_errors.py   # 错例归族分析，产出 error_analysis.json
python code/make_figures.py     # 重生 fig1-9（可加数字参数只重画单张，如 `make_figures.py 9`）
python code/qc_article.py       # 文章自检
```

数据脚本自动下载，无需手动准备。`train_cnn.py` 在 CPU 上全程 1295 秒（约 21.6 分钟）。

## 网络结构

```
输入 3×32×32
  Conv 3→16,  3×3, pad=1  →  [BN]  ReLU  MaxPool2  →  16 通道 16×16
  Conv 16→32, 3×3, pad=1  →  [BN]  ReLU  MaxPool2  →  32 通道 8×8
  Conv 32→64, 3×3, pad=1  →  [BN]  ReLU  MaxPool2  →  64 通道 4×4
展平 → 1024
  Linear 1024→64  ReLU
  Linear 64→43   Softmax + 交叉熵
```

优化器为带动量 0.9 的 SGD，批大小 64。`use_bn` 与 `channels` 都是可配置的，改一行就能换结构。

## 关键数字（均来自 stats.json）

| 项 | 值 |
|---|---|
| 数据集 | GTSRB-Training_fixed.zip，43 类，原始 26640 张（每类 150 ~ 1500，差 10 倍）|
| 切分后 | 训练 10772 / 测试 2056；训练每类 128 ~ 300（不平衡 2.34 倍）|
| 第一层卷积参数 | 448（432 权重 + 16 偏置）|
| 等价全连接参数 | 50348032 |
| 减少倍数 | 112384×（= 局部连接 109.75 × 权重共享 1024）|
| 全网络总参数 | 92203（带 BN）/ 91979（不带 BN），BN 只多 224 个 |
| 参数构成 | 卷积 23584（25.6%）+ BN 224（0.2%）+ 全连接 68395（74.2%）|
| 梯度校验 | float64 下各层相对误差 2.2e-6 ~ 5.8e-6，全部 PASS |
| 学习率扫描（6 轮 / 4000 张） | 带 BN 最优 lr=0.01 → 93.9%；不带 BN 最优 lr=0.01 → 87.9% |
| 差距随学习率扩大 | lr=0.01 差 6.0 个百分点，lr=0.05 差 23.3，lr=0.1 差 29.8 |
| 2×2 全量对照（12 轮） | A 带BN@0.01 98.74%｜B 不带BN@0.01 98.30%｜C 带BN@0.05 94.84%｜D 不带BN@0.05 3.94% |
| 收敛速度（A vs B） | 首轮 79.4% vs 63.0%；首次 ≥98% 分别在第 8 / 第 10 轮 |
| 错例结构（A 组 26 错 / D 组 1975 错） | A：同族 21、跨族 5；D：1930 张（97.7%）全塌向「限速30」，只有 5 个类有过命中 |

## 踩过的坑

- `im2col` 的通道顺序必须和权重摊平顺序一致（都是先走核的宽高位置、再走通道），否则矩阵乘法不会报错，只会让 loss 一直卡在 0.5 附近。
- 数据是 NHWC，卷积按 NCHW 实现。不转置不会立刻报错，会在矩阵乘法那一步抛 `size 288 vs 27`。
- 数值梯度校验用 float32 会把相对误差抬到 0.5%，看起来像全错。必须换 float64。
- 卷积偏置后面紧跟 BN 时，它的真梯度恒为 0（常数偏移被均值消掉），这是正确现象，不是 bug。
- 大学习率（lr=0.05）不加 BN 时，前 11 轮看着还能到 60% 出头，第 12 轮会突然塌到 3.9%：训练损失从 1.19 跳到 1.93。只看中间某一轮会误判成「还行」。
- 混淆矩阵在 98.74% 准确率下几乎全是对角线，直接用单层热力图什么也看不出来。要在对角线上另叠一层，把非对角线的格子标红，错在哪里才看得见。
- 出图脚本里 `mkdir(exist_ok=True)` 在某些沙箱环境下会对已存在目录抛 PermissionError。用 `if not p.exists(): p.mkdir(parents=True)` 更稳。
