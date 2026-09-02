# -*- coding: utf-8 -*-
"""H9 假设检验与相关性：把佛山"前13名全被淘汰"变成一道统计题
数据集: UCI Adult (Census Income, 1994 US Census)  纯 numpy/scipy/pandas
"""
import numpy as np, pandas as pd, itertools, json
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["STHeiti Medium", "Heiti SC", "PingFang SC", "Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False); plt.rcParams["font.sans-serif"] = [f]; break
    except Exception: continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"axes.spines.top":False,
    "axes.spines.right":False,"font.size":11,"axes.titlesize":13,"axes.titleweight":"bold"})
OKA = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#999999"]

R = {}   # 结果字典

# ---------------------------------------------------------------- 载入
cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"]
raw = pd.read_csv("data/adult.data", names=cols, skipinitialspace=True, na_values="?")
print("原始行数:", len(raw))
print("各列缺失:", {c:int(raw[c].isna().sum()) for c in cols if raw[c].isna().sum()>0})
df = raw.dropna(subset=["education_num","hours_per_week","income","capital_gain","age"]).reset_index(drop=True)
N = len(df)
R["N"] = N
hi = (df.income == ">50K").astype(int).values          # 高收入二分类
edn = df.education_num.values.astype(float)            # 受教育年限
hrs = df.hours_per_week.values.astype(float)           # 每周工作小时
age = df.age.values.astype(float)
cg  = df.capital_gain.values.astype(float)
print(f"分析样本 N = {N}")

# ================================================================= Q1
# 佛山季华中学：18 人进面试, 5 人进体检, 笔试前 13 名全部落选, 后 5 名全部入选
# H0: 面试结果与笔试名次无关 -> 5 个名额在 18 个名次里等可能
n_all, n_sel = 18, 5
combos = list(itertools.combinations(range(1, n_all+1), n_sel))
C = len(combos)
W = np.array([sum(c) for c in combos])                 # 入选者笔试名次之和(秩和)
w_obs = sum(range(14, 19))                             # 观测: 14+15+16+17+18
p_one = float((W >= w_obs).mean())
p_two = float(((W >= w_obs) | (W <= sum(range(1, 6)))).mean())
# 正态近似(大样本 Wilcoxon)
mu = n_sel*(n_all+1)/2
sd = np.sqrt(n_sel*(n_all-n_sel)*(n_all+1)/12)
z_apx = (w_obs - mu)/sd
p_apx = float(1 - stats.norm.cdf(z_apx))
R.update({"q1_C":C,"q1_W_obs":w_obs,"q1_mu":mu,"q1_sd":float(sd),"q1_z_apx":float(z_apx),
          "q1_p_one_exact":p_one,"q1_p_two_exact":p_two,"q1_p_apx":p_apx})
print(f"\n[Q1] C(18,5)={C}  观测秩和W={w_obs}  E[W]={mu}  sd={sd:.4f}")
print(f"     精确单侧 p={p_one:.6g} ({1/p_one:.0f} 分之一)  精确双侧 p={p_two:.6g}")
print(f"     正态近似 Z={z_apx:.4f} -> p={p_apx:.6g}  (低估 {(p_apx/p_one):.1f} 倍)")

# ================================================================= Q2
# Q2a 两比例 Z 检验: 本科及以上(edu>=13) vs 以下, 高收入占比差异
g = (edn >= 13).astype(int)
n1, n0 = int(g.sum()), int((1-g).sum())
x1, x0 = int(hi[g==1].sum()), int(hi[g==0].sum())
p1, p0 = x1/n1, x0/n0
p_pool = (x1+x0)/(n1+n0)
se_pool = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n0))
Z_prop = (p1-p0)/se_pool
p_prop_two = float(2*(1-stats.norm.cdf(abs(Z_prop))))
p_prop_one = float(1-stats.norm.cdf(abs(Z_prop)))
rr = p1/p0
odds1, odds0 = p1/(1-p1), p0/(1-p0)
R.update({"q2_n_hi":n1,"q2_n_lo":n0,"q2_x_hi":x1,"q2_x_lo":x0,"q2_p_hi":p1,"q2_p_lo":p0,
          "q2_diff":p1-p0,"q2_Z":float(Z_prop),"q2_p_two":p_prop_two,"q2_p_one":p_prop_one,
          "q2_RR":float(rr),"q2_OR":float(odds1/odds0),"q2_p_pool":p_pool})
print(f"\n[Q2a] 本科及以上 n={n1} 高收入占比={p1:.4f} ({x1} 人)")
print(f"      本科以下   n={n0} 高收入占比={p0:.4f} ({x0} 人)")
print(f"      差={p1-p0:.4f}  Z={Z_prop:.4f}  p(双侧)={p_prop_two:.3g}  RR={rr:.3f}  OR={odds1/odds0:.3f}")

# Q2b 连续变量的 Welch t 检验: 两组每周工作小时
a = hrs[g==1]; b = hrs[g==0]
m1, m0 = a.mean(), b.mean(); s1, s0 = a.std(ddof=1), b.std(ddof=1)
se_w = np.sqrt(s1**2/len(a) + s0**2/len(b))
t_w = (m1-m0)/se_w
dfw = (s1**2/len(a)+s0**2/len(b))**2 / ((s1**2/len(a))**2/(len(a)-1) + (s0**2/len(b))**2/(len(b)-1))
p_tw = float(2*(1-stats.t.cdf(abs(t_w), dfw)))
Z_lg = (m1-m0)/se_w                                  # 同一差值, 误用正态
p_Z_lg = float(2*(1-stats.norm.cdf(abs(Z_lg))))
R.update({"q2b_m_hi":float(m1),"q2b_m_lo":float(m0),"q2b_s_hi":float(s1),"q2b_s_lo":float(s0),
          "q2b_t":float(t_w),"q2b_df":float(dfw),"q2b_p_t":p_tw,"q2b_Z":float(Z_lg),"q2b_p_Z":p_Z_lg})
print(f"[Q2b] 工时 {m1:.3f} vs {m0:.3f}  t={t_w:.4f} df={dfw:.1f} p={p_tw:.3g} | 误用Z p={p_Z_lg:.3g}")

# Q2c 小样本演示: 固定种子各抽 20 人, Welch t(小df) vs 大样本 Z
rng = np.random.default_rng(42)
sa = rng.choice(a, 20, replace=False); sb = rng.choice(b, 20, replace=False)
ms1, ms0 = sa.mean(), sb.mean(); ss1, ss0 = sa.std(ddof=1), sb.std(ddof=1)
se_s = np.sqrt(ss1**2/20 + ss0**2/20); t_s = (ms1-ms0)/se_s
dfs_ = (ss1**2/20+ss0**2/20)**2/((ss1**2/20)**2/19 + (ss0**2/20)**2/19)
p_ts = float(2*(1-stats.t.cdf(abs(t_s), dfs_)))
p_zs = float(2*(1-stats.norm.cdf(abs(t_s))))
tc_005_t = float(stats.t.ppf(0.975, dfs_))
R.update({"q2c_mean_hi":float(ms1),"q2c_mean_lo":float(ms0),"q2c_diff":float(ms1-ms0),
          "q2c_t":float(t_s),"q2c_df":float(dfs_),"q2c_p_t":p_ts,"q2c_p_z":p_zs,
          "q2c_tcrit":tc_005_t})
print(f"[Q2c] 各抽20人: 差={ms1-ms0:.3f}  t={t_s:.4f} df={dfs_:.2f} p(t)={p_ts:.4f} | p(误用Z)={p_zs:.4f}")
print(f"      t 临界值(0.05,df={dfs_:.1f})={tc_005_t:.4f}  vs  正态 1.9600")

# ================================================================= Q3
# Q3a 4 档学历 x 收入 卡方独立性检验
bins   = [0, 9, 11, 13, 20]
labels = ["初中及以下(1-9)", "高中/部分大学(10-11)", "学士(13)", "硕士及以上(14-16)"]
edu_band = pd.cut(df.education_num, bins=[0,9,10,13,20], labels=labels, right=True)
ct = pd.crosstab(edu_band, df.income)
ct = ct[["<=50K", ">50K"]]
O = ct.values.astype(float)
E = np.outer(O.sum(1), O.sum(0))/O.sum()
chi2 = float(((O-E)**2/E).sum())
dfree = (O.shape[0]-1)*(O.shape[1]-1)
p_chi = float(stats.chi2.sf(chi2, dfree))
V = float(np.sqrt(chi2/(O.sum()*min(O.shape[0]-1, O.shape[1]-1))))
contrib = ((O-E)**2/E)
R.update({"q3_chi2":chi2,"q3_df":dfree,"q3_p":p_chi,"q3_V":V,
          "q3_n":int(O.sum()),"q3_rate":[float(r) for r in O[:,1]/O.sum(1)],
          "q3_nrow":[int(v) for v in O.sum(1)],"q3_maxcontrib":float(contrib.max())})
print(f"\n[Q3a] 4x2 列联表 chi2={chi2:.2f} df={dfree} p={p_chi:.3g}  Cramer's V={V:.4f}")
for lab, row, exp in zip(labels, O, E):
    print(f"      {lab:22s} n={int(row.sum()):6d} 高收入占比={row[1]/row.sum():.4f}")

# Q3b 2x2: edu>=13 x income>50K  -> phi 系数, 验证 phi^2 = chi2/n
o = np.array([[ ((g==1)&(hi==1)).sum(), ((g==1)&(hi==0)).sum() ],
               [ ((g==0)&(hi==1)).sum(), ((g==0)&(hi==0)).sum() ]], dtype=float)
e = np.outer(o.sum(1), o.sum(0))/o.sum()
chi2_22 = float(((o-e)**2/e).sum())
phi = float(np.sqrt(chi2_22/o.sum()))
phi_direct = float((o[0,0]*o[1,1]-o[0,1]*o[1,0])/np.sqrt(o[0].sum()*o[1].sum()*o[:,0].sum()*o[:,1].sum()))
R.update({"q3b_chi2":chi2_22,"q3b_phi":phi,"q3b_phi_direct":phi_direct,
          "q3b_phi2":phi**2,"q3b_chi2_over_n":chi2_22/o.sum(),"q3b_p":float(stats.chi2.sf(chi2_22,1))})
print(f"[Q3b] 2x2 chi2={chi2_22:.2f}  phi={phi:.6f} (直接算 {phi_direct:.6f})")
print(f"      phi^2={phi**2:.8f}   chi2/n={chi2_22/o.sum():.8f}   相等? {np.isclose(phi**2, chi2_22/o.sum())}")

# Q3c Cochran-Armitage 趋势检验 (有方向 -> 单侧, 比无序卡方更有力)
w = np.array([1.,2.,3.,4.])                            # 学历档得分
ni = O.sum(1); xi = O[:,1]; Nt = ni.sum(); pbar = xi.sum()/Nt
num = float((w*(xi - ni*pbar)).sum())
var = pbar*(1-pbar)*( (ni*w**2).sum() - (ni*w).sum()**2/Nt )
z_ca = num/np.sqrt(var)
p_ca_one = float(1-stats.norm.cdf(z_ca)); p_ca_two = float(2*(1-stats.norm.cdf(abs(z_ca))))
R.update({"q3c_z":float(z_ca),"q3c_p_one":p_ca_one,"q3c_p_two":p_ca_two})
print(f"[Q3c] Cochran-Armitage 趋势 Z={z_ca:.4f} 单侧 p={p_ca_one:.3g} 双侧 p={p_ca_two:.3g}")

# Q3d 效应量 vs 样本量: 同一 phi, n 变化时 chi2 与 p 的爆炸
rows=[]
for nn in [100, 500, 2000, 10000, int(o.sum())]:
    c2 = phi**2*nn
    rows.append({"n":nn, "chi2":c2, "p":float(stats.chi2.sf(c2,1))})
R["q3d"] = rows
print("[Q3d] 同一 phi=%.4f, 样本量变化时:" % phi)
for r_ in rows: print(f"      n={r_['n']:6d}  chi2={r_['chi2']:8.2f}  p={r_['p']:.3g}")

# ================================================================= Q4
# 相关系数全家桶
def pearson(x,y): return float(np.corrcoef(x,y)[0,1])
r_p = pearson(edn, hrs)
r_s = float(stats.spearmanr(edn, hrs).statistic)
r_k = float(stats.kendalltau(edn, hrs).statistic)
# 点二列: 真二分类 x 连续
r_pb = pearson(hi, edn)
r_pb_hrs = pearson(hi, hrs)
# phi: 两个二分类
R.update({"q4_pearson_edn_hrs":r_p,"q4_spearman_edn_hrs":r_s,"q4_kendall_edn_hrs":r_k,
          "q4_rpb_hi_edn":r_pb,"q4_rpb_hi_hrs":r_pb_hrs,"q4_phi":phi})
print(f"\n[Q4] education_num vs hours:  Pearson={r_p:.4f}  Spearman={r_s:.4f}  Kendall={r_k:.4f}")
print(f"     点二列 r_pb(高收入, 受教育年限)={r_pb:.4f}   r_pb(高收入, 工时)={r_pb_hrs:.4f}")

# 恒等式 1: 点二列 r_pb 与两样本 t 检验一一对应  t = r*sqrt(n-2)/sqrt(1-r^2)
t_from_r = r_pb_hrs*np.sqrt(N-2)/np.sqrt(1-r_pb_hrs**2)
tt = stats.ttest_ind(hrs[hi==1], hrs[hi==0], equal_var=True)
R.update({"q4_t_from_rpb":float(t_from_r),"q4_t_direct":float(tt.statistic),
          "q4_p_from_rpb":float(2*stats.t.sf(abs(t_from_r),N-2)),"q4_p_direct":float(tt.pvalue)})
print(f"     恒等式1: t = r_pb*sqrt(n-2)/sqrt(1-r_pb^2) = {t_from_r:.4f}   直接 t 检验 = {tt.statistic:.4f}")

# 恒等式 2: Spearman = 秩上的 Pearson
rk_edn = stats.rankdata(edn); rk_hrs = stats.rankdata(hrs)
r_s_check = pearson(rk_edn, rk_hrs)
# 恒等式 3: Kendall tau = (一致对-不一致对)/总对数  (抽样 3000 对验证, 全量 4.5 亿对太慢)
rng2 = np.random.default_rng(7)
idx = rng2.choice(N, 3000, replace=False)
pair_i, pair_j = np.triu_indices(3000, 1)
sx = edn[idx]; sy = hrs[idx]
d1 = sx[pair_i]-sx[pair_j]; d2 = sy[pair_i]-sy[pair_j]
conc = int(((d1*d2)>0).sum()); disc = int(((d1*d2)<0).sum())
tau_est = (conc-disc)/(conc+disc)
R.update({"q4_spearman_check":float(r_s_check),"q4_tau_sample":float(tau_est),
          "q4_conc":conc,"q4_disc":disc})
print(f"     恒等式2: Spearman(直接)={r_s:.6f}  Pearson(秩上)={r_s_check:.6f}")
print(f"     恒等式3: Kendall(直接)={r_k:.4f}  抽样3000人一致对/不一致对估计={tau_est:.4f} (一致{conc} 不一致{disc})")

# 极端值敏感性: Pearson vs Spearman  (与 fig4 完全一致: n=200, seed=1, 注入 3 个 [1年,99h])
rng3 = np.random.default_rng(1)
sub = rng3.choice(N, 200, replace=False)
xe, ye = edn[sub].copy(), hrs[sub].copy()
r_p0, r_s0 = pearson(xe,ye), float(stats.spearmanr(xe,ye).statistic)
# 加入 3 个真实世界里可能出现的极端观测 (1 年教育 + 每周 99 小时)
xout = np.r_[xe, [1.,1.,1.]]; yout = np.r_[ye, [99.,99.,99.]]
r_p1, r_s1 = pearson(xout,yout), float(stats.spearmanr(xout,yout).statistic)
R.update({"q4_out_rp_before":r_p0,"q4_out_rs_before":r_s0,"q4_out_rp_after":r_p1,"q4_out_rs_after":r_s1})
print(f"     极端值: Pearson {r_p0:.4f} -> {r_p1:.4f} (变 {r_p1-r_p0:+.4f}) | Spearman {r_s0:.4f} -> {r_s1:.4f} (变 {r_s1-r_s0:+.4f})")

# 单侧 vs 双侧: 扫描 14 个职业 vs 其余人 的工时差异 (多重比较演示)
occ = raw.loc[df.index, "occupation"]
res=[]
for o_ in sorted(occ.dropna().unique()):
    m = (occ==o_).values
    if m.sum() < 30: continue
    t_ = stats.ttest_ind(hrs[m], hrs[~m], equal_var=False)
    res.append({"occ":o_,"n":int(m.sum()),"diff":float(hrs[m].mean()-hrs[~m].mean()),
                "t":float(t_.statistic),"p_two":float(t_.pvalue),"p_one":float(t_.pvalue/2)})
res_df = pd.DataFrame(res)
cross = res_df[(res_df.p_two>0.05)&(res_df.p_one<0.05)]
sig_two = int((res_df.p_two<0.05).sum()); sig_one = int((res_df.p_one<0.05).sum())
n_tests = len(res_df)
bonf = 0.05/n_tests
sig_bonf = int((res_df.p_two<bonf).sum())
R.update({"q4_n_tests":n_tests,"q4_sig_two":sig_two,"q4_sig_one":sig_one,
          "q4_bonf_alpha":bonf,"q4_sig_bonf":sig_bonf,
          "q4_cross":[{"occ":r_["occ"],"n":int(r_["n"]),"diff":round(r_["diff"],3),
                       "p_two":round(r_["p_two"],4),"p_one":round(r_["p_one"],4)} for _,r_ in cross.iterrows()]})
print(f"\n[Q4] 扫描 {n_tests} 个职业的工时差异: 双侧显著 {sig_two} 个, 单侧显著 {sig_one} 个,")
print(f"     双侧>0.05 但单侧<0.05 的 '擦边' 案例 {len(cross)} 个; Bonferroni 校正 alpha={bonf:.4f} 后仍显著 {sig_bonf} 个")
for _,r_ in cross.iterrows():
    print(f"     - {r_['occ']:18s} n={int(r_['n']):5d} 差={r_['diff']:+.3f} 双侧p={r_['p_two']:.4f} 单侧p={r_['p_one']:.4f}")

# ================================================================= 输出
pd.DataFrame([R]).to_json("data/h9_results_raw.json", force_ascii=False, indent=1)
rows=[]
for k,v in R.items():
    if isinstance(v,(int,float,str)): rows.append({"key":k,"value":v})
pd.DataFrame(rows).to_csv("data/h9_results.csv", index=False)
ct.to_csv("data/h9_contingency.csv")
res_df.to_csv("data/h9_occupation_tests.csv", index=False)
print("\n已写出 data/h9_results.csv, h9_contingency.csv, h9_occupation_tests.csv")
print("CT:\n", ct)
