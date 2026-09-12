# M10 · 1460 套二手房丢进决策树：它到底按什么给房子定价

「AI观星记」系列第 M10 篇的完整可复现工程。文章正文见 [`M10_房价决策树.md`](M10_房价决策树.md)。

一句话结论：决策树给房子定价，本质是把市场按「质量、房龄、面积」切成一块块矩形，每块给一个平均价；分类树先看房龄，回归树先看质量，而树不是越深越好，修剪才是正经事。

## 快速复现

```bash
# 建议 Python 3.13；依赖见 requirements.txt
pip install -r requirements.txt

python code/experiment.py    # 清洗 + 手搓树 + 分类树 + 回归树 + 预剪枝 + 后剪枝 → stats.json
python code/figures.py       # 出图 → figures/（7 张，全部从 stats.json 重绘）
python code/qc_article.py    # 校验文章数字与 stats.json 是否一致
```

`experiment.py` 在本地找不到原始快照时会自动从 OpenML 拉取，不需要任何凭证。所有含随机性的步骤都固定了种子（`RNG = 42`），重跑 `stats.json` 可逐字段复现。

## 数据来源

| 项 | 说明 |
| --- | --- |
| 数据集 | Ames housing（艾姆斯市二手房成交记录） |
| 等价于 | Kaggle `house-prices-advanced-regression-techniques` 的 `train.csv` |
| 获取方式 | OpenML `house_prices` v1，`fetch_openml` 直连 |
| 原始规模 | 1460 行 × 80 特征 + SalePrice |
| 建模规模 | 1460 行 × 45 特征 |

清洗里做的事都写在 `experiment.py` 的注释里：数值列的 `NA` 哨兵用中位数填补（原始表里这些列因哨兵被读成 object）、ExterQual/KitchenQual 等字母评级映射成 1 到 5 的序数、Neighborhood（街区，25 个水平）做独热编码。

## 目录

```
AI观星记_M10_决策树_房价/
├── M10_房价决策树.md        # 文章正文
├── M10_运营物料.md          # 标题 / 标签 / 评论引导 / 多平台改编 / 沸点
├── stats.json               # 全部实测数值（文章所有数字的唯一来源）
├── requirements.txt
├── code/
│   ├── experiment.py        # 手搓 CART + sklearn 分类/回归树 + 预剪枝/后剪枝 → stats.json
│   ├── figures.py           # 7 张数据图，全部从 stats.json 与 data/ 重绘
│   └── qc_article.py        # 文章 QC（字数/黑名单词/破折号/粗体/数学占比/数字一致性）
├── data/
│   ├── house_prices.csv     # OpenML 原始快照 1460×81
│   ├── processed.csv        # 清洗后特征矩阵 + SalePrice + above_median
│   ├── X.npy / y.npy / above.npy
│   └── feat_names.json
├── figures/                 # 7 张数据图（matplotlib 重绘）
└── illustrations/           # 小桃 IP 封面图（ImageGen 生成）
```

## 关键结果

数据先按成交价中位数（16.3 万美元）切成两半：高于 728 套、低于或等于 732 套，几乎对半。

分类树（`criterion="entropy"`, 层数 4）的根节点选择 `YearBuilt <= 1984.5`，切完左子节点 660 套（老房）熵 0.7915、右子节点 435 套（新房）熵 0.4870。

手搓递归 CART（只用 OverallQual 与 GrLivArea，层数 3）的根节点选择 `OverallQual <= 6.5`，信息增益 0.3393，增益率 0.3554；第二层再按 `GrLivArea <= 1381.8` 切，增益 0.1876。手搓树与 sklearn 树互相印证。

12 个最强候选特征的信息增益与增益率（前 6）：

| 特征 | 信息增益 | 增益率 | 阈值 | 取值数 |
| --- | --- | --- | --- | --- |
| OverallQual | 0.3393 | 0.3554 | 6.5 | 10 |
| FullBath | 0.3128 | 0.3149 | 1.5 | 4 |
| YearBuilt | 0.3117 | 0.3188 | 1980 | 112 |
| ExterQual | 0.3027 | 0.3184 | 3.5 | 4 |
| GrLivArea | 0.2883 | 0.3183 | 1224 | 861 |
| BsmtQual | 0.2879 | 0.2879 | 3.5 | 5 |

两个分类指标的差距很小：entropy 树训练 89.2% / 测试 87.1%，gini 树训练 91.1% / 测试 86.8%。

回归树这边，过拟合是主角：

| 设置 | 训练 R² | 测试 R² | 叶子数 |
| --- | --- | --- | --- |
| 不限制层数 | 1.000 | 0.742 | 1056 |
| 层数 3 | 0.724 | 0.731 | 8 |
| 后剪枝 CCP（α=1.35e6） | 0.975 | 0.755 | 165 |

预剪枝扫描层数 1 到 15，测试 R² 在层数 3 达到峰值 0.731，之后上下波动再没超过。后剪枝把 1056 片叶子压到 165 片，测试 R² 反而从 0.742 升到 0.755。

回归树的特征重要性也说明了另一件事：`OverallQual` 高达 0.634，第二名 `GrLivArea` 只有 0.109。分类树则更看重 `YearBuilt`（0.477）。同一批房子，问「贵不贵」和问「多少钱」，树关心的东西不一样。

只用 GrLivArea 与 YearBuilt 两个特征、层数 6 的回归树，测试 R² 就能达到 0.772，比全特征层数 3 的 0.731 还高。面积和年份本身已经承载了房价的大部分信息。

## 环境

Python 3.13.12，numpy 2.5.1，pandas 3.0.5，matplotlib 3.11.1，scikit-learn 1.9.0。

配图用 Okabe-Ito 色盲安全配色、PingFang SC 中文字体，`figure.dpi=150` / `savefig.dpi=300`。
