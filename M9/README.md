# M9 · 3104 款手机规格的无监督聚类（KMeans + DBSCAN）

「AI观星记」系列第 M9 篇的完整可复现工程。文章正文见 [`M9_手机规格聚类.md`](M9_手机规格聚类.md)。

一句话结论：手机规格数据能按参数聚成四档，但这四档的价格区间大幅重叠，规格分层和价格分层是两件事；而折叠屏这类机型在任何一档里都放不下，DBSCAN 把它们单独标成了噪声。

## 快速复现

```bash
# 建议 Python 3.13；依赖见 requirements.txt
pip install -r requirements.txt

python code/prepare_data.py     # 清洗 → data/phone_clean.csv（3104 款，零缺失）
python code/experiment.py       # 聚类 → stats.json + data/phone_clustered.csv
python code/make_figures.py     # 出图 → figures/（5 张）
python code/qc_article.py       # 校验文章数字与 stats.json 是否一致
```

`prepare_data.py` 在本地找不到原始快照时会自动从公开仓库下载，不需要任何凭证。

## 数据来源

| 项 | 说明 |
| --- | --- |
| 数据集 | GSMArena 手机规格快照（2023-01-04 抓取） |
| 仓库 | `AlbertHunduza/Smartphone-Project` 的 `GSMArena.csv` |
| 直链 | https://raw.githubusercontent.com/AlbertHunduza/Smartphone-Project/main/GSMArena.csv |
| 原始规模 | 11936 行 × 50 列 |
| 建模规模 | 3104 款（2015–2022，八个数值特征零缺失） |

清洗里丢掉的东西都记在 `prepare_data.py` 的注释里：价格字段九成写作 `About 200 EUR` 而非带 € 符号（覆盖率因此从 5.7% 提到 66.8%）、品牌字段带 `\n1348 devices` 尾缀、以及混进来的平板（iPad mini、Galaxy Tab）按屏幕与重量上限剔除。

## 目录

```
AI观星记_M9_无监督聚类_手机规格/
├── M9_手机规格聚类.md      # 文章正文
├── stats.json              # 全部实测数值（文章所有数字的唯一来源）
├── requirements.txt
├── code/
│   ├── prepare_data.py     # 清洗 + 特征工程 → phone_clean.csv
│   ├── experiment.py       # Gap / KMeans / DBSCAN / 鲁棒性对照 → stats.json
│   ├── make_figures.py     # 5 张数据图，全部从 stats.json 重绘
│   ├── qc_article.py       # 文章 QC（字数/黑名单词/数字一致性）
│   ├── inspect_data.py     # 勘查脚本（原始表头与字段可用率）
│   ├── inspect_data2.py    # 勘查脚本（2023 快照字段）
│   └── probe_sample.py     # 勘查脚本（折叠屏/三防机在不在样本里）
├── data/
│   ├── gsmarena_raw.csv    # 原始快照 11936×50（12 MB）
│   ├── phone_clean.csv     # 清洗后 3104×20
│   ├── phone_clustered.csv # 带 KMeans 与 DBSCAN 标签
│   ├── Z.npy / PC.npy      # 标准化特征 / 主成分得分
│   └── labels_km.npy / labels_db.npy
├── figures/                # 5 张数据图（matplotlib 重绘）
└── illustrations/          # 4 张小桃 IP 原理图（ImageGen 生成）
```

## 关键结果

选 K 这件事上，两个教科书指标打架：肘部法给 K=4，轮廓系数最高在 K=2，而轮廓系数全程不超过 0.37（属于结构偏弱）。

Gap 统计量揭示了更根本的问题，同一批数据、同一个公式，只换参照系：

| 参照系 | 选出的 K |
| --- | --- |
| 逐维均匀 | 选不出（gap 单调上升，无拐点） |
| 沿主成分均匀 | 选不出（同上） |
| 多元高斯（保住均值与协方差） | K=4 |

DBSCAN 在「前 3 主成分（白化）」空间、eps=0.416、min_samples=6 下判出 4 簇、167 个噪声点（5.4%）。噪声里折叠屏 13/27、三防巨电池机 44/101，两类合计 57 款。随机抽同样数量的机型期望只命中 6.9 款，富集 8.28 倍，超几何检验 p = 1.2e-41。

同一个 eps 与 min_samples 直接用在 8 维标准化空间，噪声率跳到 70.4%（2186 款），这是高维密度退化的直接证据。

四档画像（中位数）：

| 档位 | 机型数 | 价格(€) | 屏幕(in) | 重量(g) | 电池(mAh) | 内存(GB) | 存储(GB) | ppi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 小屏入门 | 751 | 130 | 5.13 | 149 | 2611 | 1.83 | 20 | 270 |
| 大屏低价 | 612 | 146 | 6.42 | 203 | 4726 | 4.03 | 77 | 275 |
| 小屏高清 | 703 | 280 | 5.70 | 167 | 3331 | 4.41 | 87 | 429 |
| 旗舰堆料 | 1038 | 350 | 6.59 | 199 | 4615 | 9.11 | 261 | 407 |

相邻档价格重叠宽度 160 / 124.5 / 419 欧元。第三组最夸张：小屏高清档的 95 分位价（589 €）远高于旗舰堆料档的 5 分位价（170 €）。

## 环境

Python 3.13.12，numpy 2.5.1，pandas 3.0.5，matplotlib 3.11.1，scikit-learn 1.9.0，scipy 1.18.1。

所有含随机性的步骤都固定了种子（`RNG = 42`），从零重跑 `stats.json` 可逐字段复现。
