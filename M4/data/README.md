# 数据说明：online_shopping_10_cats

- 来源：ChineseNlpCorpus（SophonPlus 整理），京东商城购物评论
- 原始地址：https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip
- 规模：62,774 条评论，10 个品类（书籍/平板/手机/水果/洗发水/热水器/蒙牛/衣服/计算机/酒店），好评 31,728 / 差评 31,046
- 字段：`cat`（品类）、`label`（1=好评，0=差评）、`review`（评论文本，UTF-8）
- 本目录内为完整 CSV（约 11 MB），可直接复跑 `../code/model_word2vec_rnn.py`，一次运行产出 `../stats.json` 与 `../figures/` 全部 8 张图
- 依赖：python 3.10+，numpy / matplotlib / scikit-learn / jieba
