# M14 · 卷积层构造与 BatchNormalization 实战

配套文章：`M14_CNN_人脸_卷积与BN_2026-09-16.md`

用一份真实人脸数据集（Olivetti Faces，40 人 400 张 64×64 灰度图），从一个纯 numpy 手写的 CNN 出发，讲清三件事：卷积层的局部感受野与权值共享、卷积相对全连接的参数优势、以及 BatchNormalization 如何稳住训练。

## 目录结构

```
M14_CNN_batch_norm/
├── M14_CNN_人脸_卷积与BN_2026-09-16.md   # 正文
├── code/
│   ├── train_cnn.py       # 从零 numpy CNN：卷积(im2col)/池化/BN/全连接 + 训练
│   ├── make_figures.py    # 从 stats.json + plot_data.npz 重生四张配图
│   └── qc_article.py      # 文章 QC：字数/黑名单/数学占比/数字一致性
├── data/
│   ├── stats.json         # 全文所有数字的唯一来源
│   ├── plot_data.npz      # 出图所需的中间量（loss 曲线/特征图/混淆矩阵）
│   ├── run.log            # 训练日志
│   └── olivetti_py3.pkz   # Olivetti 数据集缓存（sklearn 自动下载）
├── figures/               # fig1-4
└── requirements.txt
```

## 复现

```bash
pip install -r requirements.txt
python code/train_cnn.py      # 训练，CPU 约 5 分钟；产出 stats.json + plot_data.npz
python code/make_figures.py   # 由产物重生成 figures/fig1-4
python code/qc_article.py     # 文章自检
```

数据集无需手动下载，`fetch_olivetti_faces` 会自动从 scikit-learn 拉取（约 1.4 MB）。

## 关键数字（均来自 stats.json）

| 项 | 值 |
|---|---|
| 任务 | 前 20 人 vs 后 20 人（300 训练 / 100 测试） |
| 第一层卷积参数 | 88（80 权重 + 8 偏置） |
| 等价全连接参数 | 32776 |
| 减少倍数 | 409.7×（纯权重口径） |
| 全网络总参数 | 9442 |
| 带 BN（lr=0.01） | 准确率 0.95，终 loss 0.0033，148.7 s |
| 不带 BN（lr=0.005） | 准确率 0.65，loss 卡在 0.3464，151.1 s |
| 不带 BN（lr=0.01） | 准确率 0.78，未爆炸 |

## 网络结构

`Conv(3×3, 8) → ReLU → MaxPool2 → [BN] → Conv(3×3, 16) → ReLU → MaxPool2 → [BN] → FC(4096 → 2)`

无 torch / TensorFlow 依赖，卷积、池化、BN 的前向与反向均为 numpy 手写，便于对照公式逐步阅读。
