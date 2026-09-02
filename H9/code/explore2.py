import numpy as np, pandas as pd
from scipy import stats
cols = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation",
        "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
df = pd.read_csv("data/adult.data", names=cols, skipinitialspace=True, na_values="?")
hi = (df.income==">50K").astype(int).values; edn=df.education_num.values.astype(float)
hrs=df.hours_per_week.values.astype(float); cg=df.capital_gain.values.astype(float)
P=lambda x,y: float(np.corrcoef(x,y)[0,1])

print("=== p 值量级 (Z=59.0365) ===")
from scipy.stats import norm
z=59.0365
# log10 of two-sided p: log10(2) + log10(sf(z));  sf(z) ~ pdf(z)/z
lp = np.log10(2) + (-0.5*np.log(2*np.pi) - z**2/2)/np.log(10) - np.log10(z)
print(f"Z=59.0365 双侧 p 约 10^{lp:.1f}")
z2=59.8405
lp2 = np.log10(2) + (-0.5*np.log(2*np.pi) - z2**2/2)/np.log(10) - np.log10(z2)
print(f"CA趋势 Z=59.8405 双侧 p 约 10^{lp2:.1f}")
print("t=25.4936 p:", stats.t.sf(25.4936,13806.1)*2)
print("t=42.5839 p:", stats.t.sf(42.5839,32559)*2)
import math
def t_logp(t,df):
    # 大df下 t 近似正态
    return np.log10(2)+(-0.5*np.log(2*np.pi)-t**2/2)/np.log(10)-np.log10(t)
print(f"t=25.4936 双侧 p 约 10^{t_logp(25.4936,13806):.1f}")
print(f"t=42.5839 双侧 p 约 10^{t_logp(42.5839,32559):.1f}")

print("\n=== 真实极端值(长工时子群)对 Pearson/Spearman 的影响 ===")
for cut in [99, 90, 85, 80]:
    m = hrs <= cut
    print(f"  去掉 hours>{cut} (共{(~m).sum()}人, 占{(~m).mean()*100:.2f}%): "
          f"Pearson={P(edn[m],hrs[m]):.4f} ({P(edn[m],hrs[m])-P(edn,hrs):+.4f})  "
          f"Spearman={stats.spearmanr(edn[m],hrs[m]).statistic:.4f} "
          f"({stats.spearmanr(edn[m],hrs[m]).statistic-stats.spearmanr(edn,hrs).statistic:+.4f})")

print("\n=== capital_gain 重尾: 去掉 top-k 个极端值 ===")
print(f"  全量: Pearson={P(edn,cg):.4f}  Spearman={stats.spearmanr(edn,cg).statistic:.4f}")
for k in [1,5,10,50]:
    idx = np.argsort(cg)[:-k]
    print(f"  去掉 top{k}: Pearson={P(edn[idx],cg[idx]):.4f} ({P(edn[idx],cg[idx])-P(edn,cg):+.4f})  "
          f"Spearman={stats.spearmanr(edn[idx],cg[idx]).statistic:.4f} "
          f"({stats.spearmanr(edn[idx],cg[idx]).statistic-stats.spearmanr(edn,cg).statistic:+.4f})")

print("\n=== 小样本加极端点演示 (n=200) ===")
rng=np.random.default_rng(1)
sub=rng.choice(len(df),200,replace=False)
xe,ye=edn[sub].copy(),hrs[sub].copy()
p0,s0=P(xe,ye),float(stats.spearmanr(xe,ye).statistic)
for k in [1,3]:
    xo=np.r_[xe,[1.]*k]; yo=np.r_[ye,[99.]*k]
    p1,s1=P(xo,yo),float(stats.spearmanr(xo,yo).statistic)
    print(f"  n=200 加{k}个点: Pearson {p0:.4f}->{p1:.4f} ({p1-p0:+.4f}, {(p1-p0)/abs(p0)*100:+.1f}%) | "
          f"Spearman {s0:.4f}->{s1:.4f} ({s1-s0:+.4f}, {(s1-s0)/abs(s0)*100:+.1f}%)")

print("\n=== Kendall tau 与 Spearman 的关系 (education_num 有大量并列) ===")
print("education_num 唯一值个数:", len(np.unique(edn)), " N=", len(df))
print("hours 唯一值个数:", len(np.unique(hrs)))
