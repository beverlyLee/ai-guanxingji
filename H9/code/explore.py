import pandas as pd, numpy as np
cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"]
df = pd.read_csv("data/adult.data", names=cols, skipinitialspace=True, na_values="?")
df = df.dropna().reset_index(drop=True)
print("N =", len(df))
print("\n=== education_num 分布 ===")
print(df.education_num.value_counts().sort_index())
print("\n=== income ===")
print(df.income.value_counts())
print("\n=== hours_per_week ===")
print(df.hours_per_week.describe())
print("\n=== capital_gain ===")
print("零占比:", (df.capital_gain==0).mean().round(4))
print(df.capital_gain.describe())
print("\n=== age ===")
print(df.age.describe())
print("\n=== 相关对比 (education_num, hours_per_week) ===")
from scipy import stats
x, y = df.education_num.values, df.hours_per_week.values
print("pearson", np.corrcoef(x,y)[0,1].round(4))
print("spearman", stats.spearmanr(x,y).statistic.round(4))
print("kendall", stats.kendalltau(x,y).statistic.round(4))
print("\n=== 相关对比 (capital_gain, hours_per_week) ===")
cg = df.capital_gain.values
print("pearson", np.corrcoef(cg,y)[0,1].round(4))
print("spearman", stats.spearmanr(cg,y).statistic.round(4))
print("kendall", stats.kendalltau(cg,y).statistic.round(4))
print("\n=== 相关对比 (age, hours_per_week) ===")
a = df.age.values
print("pearson", np.corrcoef(a,y)[0,1].round(4))
print("spearman", stats.spearmanr(a,y).statistic.round(4))
print("\n=== 相关对比 (education_num, capital_gain) ===")
print("pearson", np.corrcoef(x,cg)[0,1].round(4))
print("spearman", stats.spearmanr(x,cg).statistic.round(4))
print("\n=== 职业样本量 ===")
print(df.occupation.value_counts())
print("\n=== 分组 hours 均值 (edu>=13 vs <13) ===")
g1 = df[df.education_num>=13].hours_per_week
g2 = df[df.education_num<13].hours_per_week
print("n1",len(g1),"mean",g1.mean().round(4),"std",g1.std(ddof=1).round(4))
print("n2",len(g2),"mean",g2.mean().round(4),"std",g2.std(ddof=1).round(4))
