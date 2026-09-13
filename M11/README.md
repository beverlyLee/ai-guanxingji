# M11 随机森林：体检报告箭头背后的集成学习

本目录是「AI观星记」M11 篇的完整复现包：用 UCI 克利夫兰心脏病数据集演示随机森林、提升算法与 Stacking。

## 数据

- UCI Cleveland (OpenML `cleveland` v1)
- 303 位病人，13 项临床指标
- 6 处缺失（ca 4 处、thal 2 处），以众数填补
- 二分类目标：>0 视为患病，患病比例 45.9%

## 文件

```
.
├── M11_随机森林_心脏病.md   # 正文
├── code/
│   ├── experiment.py        # 生成 stats.json 与处理后数据
│   ├── figures.py           # 从 stats.json 重生 7 张配图
│   └── qc_article.py        # 文章风格与数字一致性校验
├── data/
│   ├── processed.csv        # 填补后的数据
│   ├── X.npy / y.npy        # 数值化数组
│   └── feat_names.json      # 特征名
├── figures/                 # 7 张配图
├── stats.json               # 文章所有数字的唯一来源
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

- 单棵深树在训练集上达到 100%，5 折 CV 仅 76.5%，过拟合明显。
- Bagging 将深树的 CV 准确率提升到 81.2%，标准差下降。
- 随机森林 OOB 准确率 83.2%，CV 准确率 82.2%。
- AdaBoost CV 82.8%、GradientBoosting 79.5%、Stacking 81.8%，与 RF 处于同一区间。
- 特征重要性前五位：cp、ca、thalach、thal、oldpeak，与心内科常规评估项一致。

> 本文所有数字均来自 `stats.json`，配图由 `figures.py` 从同一文件重生。
