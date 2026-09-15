# M13 提前一天预报雾霾：手搓神经网络，讲清学习率、初始化、正则

「AI观星记」M 系列第 13 篇。用公开的 UCI 北京空气质量数据，从零用 NumPy 搭一个 MLP，
提前 24 小时预报 PM2.5，把神经网络训练里的几个关键旋钮逐个拆开讲透。

## 数据集

- UCI Beijing Multi-Site Air-Quality Data，编号 501
- 下载地址：https://archive.ics.uci.edu/static/public/501/beijing+multi-site+air+quality+data.zip
- 取其中古城站（Gucheng）单站，逐小时，2013-03-01 至 2017-02-28，共 35064 条
- 用到的特征：PM2.5、PM10、TEMP、PRES、DEWP、WSPM、RAIN
- 数据不随仓库提交，`experiment.py` 首次运行会自动下载并解开嵌套压缩包

## 如何跑

```bash
pip install -r requirements.txt
python experiment.py   # 跑全部实验，产出 stats.json 与 curves.npz，约 1 分钟
python figures.py      # 由 stats.json 与原始数据重画全部配图
python qc_article.py   # 文章自检：字数/黑名单/公式占比/数字一致性等八项
```

## 文件

| 文件 | 说明 |
| --- | --- |
| `M13_提前一天预报雾霾_2026-09-15.md` | 正文 |
| `experiment.py` | 从零实现的 MLP 与全部对比实验，文章所有数字的唯一来源 |
| `stats.json` | 实验结果，正文引用的每个数字都能在这里找到 |
| `figures.py` | 配图脚本 |
| `figures/` | 7 张配图 |
| `qc_article.py` | 文章自检脚本 |
| `M13_选题方案.md` | 选题与冲突核查记录 |

## 主要结论

- 提前 1 小时：持续性基线 R² 0.9482，神经网络 0.9241，基线赢，说明这个时距不值得上模型
- 提前 24 小时：持续性基线 R² 掉到 -0.0984，神经网络 0.2587，模型才有价值
- 学习率：1e-2 是甜点，0.3 与 1.0 一轮就发散成 NaN
- 初始化：五层网里 std=0.01 逐层坍缩到 0，Xavier/He 稳定在 1 附近，std=0.5 饱和
- dropout 0.2 把训练与验证的损失差从 0.3502 收到 0.1439
- L2 把权重范数从 10.24 收到 3.79，验证集 R² 从 0.137 升到 0.2354
- 最终配置测试集：RMSE 86.02，MAE 57.09，R² 0.2587
