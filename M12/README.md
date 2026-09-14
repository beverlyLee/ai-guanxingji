# M12 支持向量机：诈骗短信拦截背后的那条最宽分界线

本目录是「AI观星记」M12 篇的完整复现包：用 UCI Spambase 数据集演示支持向量机、软间隔 C、拉格朗日对偶与核函数。

## 数据

- UCI Spambase (OpenML `spambase` v1)
- 4601 封邮件，57 个数值特征
- 正常邮件 2788 封，诈骗邮件 1813 封，诈骗比例 39.4%
- 57 个特征 = 48 个词频 + 6 个字符频率 + 3 个大写连续长度统计
- 特征已标准化，适合 SVM 训练

## 文件

```
.
├── M12_支持向量机_诈骗短信.md   # 正文
├── code/
│   ├── experiment.py            # 生成 stats.json 与处理后数据
│   ├── figures.py               # 从 stats.json 重生 7 张配图
│   └── qc_article.py            # 文章风格与数字一致性校验
├── data/
│   ├── processed.csv            # 原始数据 + 标签
│   ├── X_raw.npy / X_scaled.npy # 标准化前后特征
│   ├── y.npy                    # 标签
│   ├── feat_names.json          # 特征名
│   └── pca_*.npy                # 2D PCA 投影与决策边界网格
├── figures/                     # 7 张配图
├── stats.json                   # 文章所有数字的唯一来源
└── requirements.txt
```

## 复现

```bash
# 建议创建虚拟环境
pip install -r requirements.txt

python code/experiment.py   # 生成 stats.json + data/
python code/figures.py      # 重生配图
python code/qc_article.py   # 校验文章
```

随机种子固定为 42，结果可复现。

## 主要结论

- Spambase 里区分度最高的三个特征全是大写连续长度统计，诈骗邮件偏好全大写吼叫。
- 线性 SVM C=1 时 5 折交叉验证准确率 92.6%，支持向量 934 个（占 20.3%），最大间隔宽度 0.384。
- C 从 0.01 增加到 100，间隔宽度从 1.205 降到 0.075，支持向量从 1326 降到 903，交叉验证准确率稳定在 91.9%–92.7%。
- 核函数对比：RBF 核 93.2% 略胜线性核 92.6%，多项式核 77.8% 明显过拟合。
- 2 维 PCA 仅解释 17.3% 方差，真实决策边界仍需在 57 维空间理解。

> 本文所有数字均来自 `stats.json`，配图由 `figures.py` 从同一文件重生。
