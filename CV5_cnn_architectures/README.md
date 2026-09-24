# CV5 · 从手写数字到ImageNet冠军：CNN架构20年，为什么越深越难训？

配套文章：`正文.md`

按时间线拆解五代卷积网络：LeNet-5（1998）、AlexNet（2012）、VGG16（2014）、GoogLeNet（2014）、ResNet-50（2015）。这篇文章不跑训练，而是从每层的架构定义出发，手算参数量、乘加次数（MACs）和感受野，再把算出来的数字和论文公开值做对照，讲清每一代的取舍：为什么 VGG16 有 1.38 亿参数，ResNet-50 只有 2555 万却更准；为什么三个 3×3 等效一个 7×7 却更省；为什么网络越深反而越难训，残差连接又是怎么解决的。

## 目录结构

```
CV5_cnn_architectures/
├── 正文.md                # 正文
├── stats.json             # 全文所有数字的唯一来源
├── code/
│   ├── experiment.py      # 从架构定义手算 params/MACs/感受野，产出 stats.json
│   ├── figures.py         # 从 stats.json 重生 fig1-6
│   └── qc_article.py      # 文章 QC：字数/黑名单/破折号/粗体/数学占比/数字一致性/图片数
├── figures/               # fig1-6
└── requirements.txt
```

## 六张配图

| 图 | 文件 | 讲一件事 |
|---|---|---|
| 图 1 | fig1_macro.png | 宏观：参数怎么先爆炸（VGG 1.38 亿）再收敛，top-1 却一路向上 |
| 图 2 | fig2_lenet.png | LeNet-5 结构：CNN 的原始模板 |
| 图 3 | fig3_vgg_stack.png | 三个 3×3 等效一个 7×7，但参数更省 |
| 图 4 | fig4_inception.png | Inception 多分支 + 1×1 瓶颈 |
| 图 5 | fig5_resnet.png | 残差块结构 + 退化曲线（plain 越深越差，ResNet 越深越好）|
| 图 6 | fig6_rf.png | 感受野随层增长：VGG16 的 212 vs ResNet-50 的 427 |

## 复现

```bash
pip install -r requirements.txt
cd code
python experiment.py   # 产出 ../stats.json
python figures.py      # 产出 ../figures/*.png
python qc_article.py   # 文章自检
```

不需要下载任何数据集，也不需要安装深度学习框架。所有数字由 `experiment.py` 从架构定义逐层累加得到，`figures.py` 只读 `stats.json` 出图。

## 关键数字（均来自 stats.json）

| 网络 | 年份 | 参数量 | MACs | ImageNet top-1 |
|---|---|---|---|---|
| LeNet-5 | 1998 | 61,706（0.0617M）| 416,520 | — |
| AlexNet | 2012 | 60,965,224（60.9652M）| 724,406,816（0.7244G）| 62.9% |
| VGG16 | 2014 | 138,357,544（138.3575M）| 15,470,264,320（15.4703G）| 71.5% |
| GoogLeNet | 2014 | 6,998,216（6.9982M）| 1,534,464,704（1.5345G）| 69.3% |
| ResNet-50 | 2015 | 25,557,032（25.557M）| 3,729,522,688（3.7295G）| 75.3% |

| 项 | 值 |
|---|---|
| 感受野（VGG16） | stage 池化后 6 / 16 / 40 / 76，末层 212 |
| 感受野（ResNet-50） | 末层 427 |
| 退化（CIFAR-10 训练误差） | plain 20 层 7.24% / plain 56 层 9.97% / ResNet 20 层 7.05% / ResNet 56 层 6.41% |
| ResNet-50 卷积 vs 全连接 | 25.557M 参数，带 BN 无偏置，每个卷积层多 2×c_out 的 BN 参数 |

## 踩过的坑

- 不同网络的卷积层参数结构不一样：LeNet/AlexNet/VGG 带偏置无 BN，ResNet 带 BN 无偏置。脚本里若用同一个默认值处理所有网络，ResNet 的参数量就会算错（曾算出 28,385,256，正确值是 25,557,032）。
- 残差块的 shortcut 分支不能直接复用主路的卷积函数，否则会改变主路尺寸造成双重下采样，网络会过早坍缩到 1×1，MACs 也会偏低。本文用一个独立的 `add_conv_params()` 只累加参数、不动尺寸。
- GoogLeNet 的参数量引论文公开值（Szegedy et al. 2015），不逐层重算。原因是原论文的 Inception 结构里有较多工程细节（如辅助分类器、逐个模块的通道配置），逐层手算反而容易和公开值偏离。
- matplotlib 画中文要在 `rcParams` 里把 `font.sans-serif` 设成 `STHeiti` 并关掉 `axes.unicode_minus`，否则负号会变成方块。图一律 `figure.dpi=150` / `savefig.dpi=300`。
