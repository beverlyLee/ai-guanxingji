# -*- coding: utf-8 -*-
"""
AI观星记 H7 · 2026 世界杯 × 概率分布
真实公开数据：mominullptr/FIFA-World-Cup-2026-Dataset (GitHub, 经 jsDelivr 镜像下载)
覆盖知识点：泊松 / 二项 / 正态(中心极限) / 卡方 / Beta(共轭) / 均匀(零模型) / 核密度(KDE)
所有数字均由本脚本从 CSV 实跑复现；配图由 matplotlib 从同 CSV 重生。
"""
import csv, math
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- 中文字体（STHeiti / 苹方回退） ----
CJK = None
for cand in ["STHeiti", "PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS"]:
    try:
        if any(cand.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            CJK = cand; break
    except Exception:
        pass
if CJK is None:
    CJK = "DejaVu Sans"
plt.rcParams["font.family"] = CJK
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# Okabe-Ito 色盲安全配色
C = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
     "purple":"#CC79A7","grey":"#999999","yellow":"#F0E442","black":"#000000"}

# ============================ 载入数据 ============================
matches = []
with open("data/matches_detailed.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        matches.append(r)
events = []
with open("data/match_events.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        events.append(r)

N = len(matches)
print(f"[INFO] 比赛场数 N = {N}")

def gi(r, k):  # 安全转 int
    v = r.get(k, "")
    return int(v) if v not in (None, "") else 0

# 常规时间总进球 / 净胜球
total_goals = [gi(r,"home_score")+gi(r,"away_score") for r in matches]
goal_diff   = [gi(r,"home_score")-gi(r,"away_score") for r in matches]
tg = np.array(total_goals, float)
gd = np.array(goal_diff, float)
lambda_hat = tg.mean()
print(f"[Q1 泊松] 总进球均值 lambda_hat = {lambda_hat:.3f}，方差 = {tg.var(ddof=1):.3f}")

# 决赛 / 季军赛
for r in matches:
    if r["stage_name"] == "Final":
        print(f"[决赛] {r['home_team_name']} {r['home_score']}-{r['away_score']} {r['away_team_name']} (result_type={r['result_type']})")
    if "Third" in r["stage_name"] or "3rd" in r["stage_name"]:
        print(f"[季军] {r['home_team_name']} {r['home_score']}-{r['away_score']} {r['away_team_name']}")

# ============================ Q1 泊松 + 卡方拟合优度 ============================
# 观测：每场总进球数分布
obs = {}
for g in total_goals:
    obs[g] = obs.get(g, 0) + 1
maxg = max(obs)
# 合并尾端使期望 >=5
bins = list(range(0, maxg+1))
exp = [N * stats.poisson.pmf(k, lambda_hat) for k in bins]
# 自底向上合并尾端
while len(bins) > 2 and exp[-1] < 5:
    bins[-2] = (bins[-2], bins[-1]); exp[-2] += exp[-1]; del bins[-1]; del exp[-1]
# 分类标签
def label(b):
    return f"≥{b[0]}" if isinstance(b, tuple) else str(b)
obs_c = [sum(obs.get(k,0) for k in (b if isinstance(b,tuple) else [b])) for b in bins]
chi2 = sum((o-e)**2/e for o,e in zip(obs_c, exp))
dof = len(bins) - 1 - 1  # 估计 lambda 减 1 自由度
p_poisson = stats.chi2.sf(chi2, dof)
print(f"[Q1 卡方] 分箱={[label(b) for b in bins]}")
print(f"[Q1 卡方] 观测={obs_c}  期望={[round(e,1) for e in exp]}")
print(f"[Q1 卡方] chi2={chi2:.2f}, dof={dof}, p={p_poisson:.3f}")

# ============================ Q2 正态(净胜球) + 中心极限 ============================
m_gd, s_gd = gd.mean(), gd.std(ddof=1)
sk = stats.skew(gd); ku = stats.kurtosis(gd)
print(f"[Q2 正态] 净胜球 mean={m_gd:.3f}, std={s_gd:.3f}, skew={sk:.3f}, kurt={ku:.3f}")
# 正态性检验(Shapiro 样本>50 用 D'Agostino)
da = stats.normaltest(gd)
print(f"[Q2 正态] D'Agostino K2 p={da.pvalue:.3f}  (p>0.05 不能拒绝正态)")

# ============================ Q3 二项(单队胜负) + Beta(共轭) ============================
# 每队每场结果
from collections import defaultdict
team_res = defaultdict(lambda: [0,0,0])  # win draw loss
for r in matches:
    hs, as_ = gi(r,"home_score"), gi(r,"away_score")
    for team, sc, opp in [(r["home_team_name"],hs,as_),(r["away_team_name"],as_,hs)]:
        if sc>opp: team_res[team][0]+=1
        elif sc==opp: team_res[team][1]+=1
        else: team_res[team][2]+=1
# 选强队与弱队示例
def team_record(name):
    w,d,l = team_res[name]; n=w+d+l
    p_win = w/(w+l) if (w+l)>0 else 0
    return w,d,l,n,p_win
for nm in ["Spain","Argentina","England","France","Mexico","South Africa"]:
    if nm in team_res:
        w,d,l,n,pw = team_record(nm)
        # Beta 共轭：均匀先验 Beta(1,1) -> Beta(1+w,1+l)
        a,b = 1+w, 1+l
        lo, hi = stats.beta.ppf([0.025,0.975], a, b)
        print(f"[Q3 Beta] {nm}: W/D/L={w}/{d}/{l} n={n} p_win={pw:.3f} | Beta(α={a},β={b}) 95%CI=[{lo:.2f},{hi:.2f}]")

# ============================ Q4 均匀(进球时刻) + 卡方 ============================
goals_min = [gi(e,"minute") for e in events if e["event_type"] in ("Goal","Own Goal")]
goals_min = [m for m in goals_min if m>0]
print(f"[Q4 均匀] 总进球事件数 = {len(goals_min)}")
# 7 个时段：0-15,...,75-90, 90+
edges = [0,15,30,45,60,75,90,1000]
labels_t = ["0-15","15-30","30-45","45-60","60-75","75-90","90+"]
bucket = np.digitize(goals_min, edges[1:-1])  # 0..5 -> 前6段, 6 -> 90+
counts_t = np.bincount(bucket, minlength=7)
exp_u = len(goals_min)/7.0
chi2_u = np.sum((counts_t-exp_u)**2/exp_u)
p_u = stats.chi2.sf(chi2_u, 6)
print(f"[Q4 均匀] 各时段进球数={list(counts_t)}")
print(f"[Q4 均匀] 均匀期望={exp_u:.1f}/段, chi2={chi2_u:.2f}, dof=6, p={p_u:.2e}")

# ============================ 图 1：泊松 ============================
fig, ax = plt.subplots(figsize=(7,4.2))
ax.bar([label(b) for b in bins], obs_c, color=C["blue"], alpha=.85, label="实际场次")
x = np.arange(0, maxg+1)
ax.plot(x, [N*stats.poisson.pmf(k, lambda_hat) for k in x], "o-", color=C["orange"], lw=2, label=f"泊松(λ={lambda_hat:.2f})")
ax.set_xlabel("单场总进球数"); ax.set_ylabel("场次数")
ax.set_title(f"单场进球数 ≈ 泊松分布 (N={N}场)")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig_goals_poisson.png"); plt.close(fig)

# ============================ 图 2：净胜球 正态 + KDE ============================
fig, ax = plt.subplots(figsize=(7,4.2))
ax.hist(gd, bins=np.arange(-8.5,8.6,1), density=True, color=C["green"], alpha=.5, label="净胜球直方图")
xs = np.linspace(-8,8,400)
ax.plot(xs, stats.norm.pdf(xs, m_gd, s_gd), "-", color=C["red"], lw=2, label=f"正态拟合 μ={m_gd:.2f},σ={s_gd:.2f}")
kde = gaussian_kde(gd)
ax.plot(xs, kde(xs), "--", color=C["purple"], lw=2, label="核密度(KDE)")
ax.set_xlabel("净胜球 = 主队进球 − 客队进球"); ax.set_ylabel("密度")
ax.set_title("净胜球 ≈ 正态，KDE 还原真实形状")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig_goaldiff_normal.png"); plt.close(fig)

# ============================ 图 3：Beta 共轭 ============================
fig, ax = plt.subplots(figsize=(7,4.2))
xsb = np.linspace(0,1,400)
for nm, col in [("Spain",C["blue"]),("Argentina",C["orange"]),("England",C["green"]),("Mexico",C["red"])]:
    if nm in team_res:
        w,d,l,n,pw = team_record(nm)
        a,b = 1+w, 1+l
        ax.plot(xsb, stats.beta.pdf(xsb,a,b), lw=2, color=col, label=f"{nm} (胜{a-1}负{b-1})")
ax.set_xlabel("该队单场胜率 p 的后验概率"); ax.set_ylabel("密度")
ax.set_title("球队胜率：Beta 共轭后验 (均匀先验)")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig_team_beta.png"); plt.close(fig)

# ============================ 图 4：均匀零模型 ============================
fig, ax = plt.subplots(figsize=(7,4.2))
ax.bar(labels_t, counts_t, color=C["purple"], alpha=.85, label="实际进球时段分布")
ax.axhline(exp_u, color=C["red"], ls="--", lw=2, label=f"均匀分布期望 ({exp_u:.1f})")
ax.set_xlabel("比赛时段(分钟)"); ax.set_ylabel("进球数")
ax.set_title(f"进球时刻 ≈ 均匀分布 (χ²={chi2_u:.1f}, p={p_u:.2f})")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig_goalminute_uniform.png"); plt.close(fig)

print("[DONE] 4 张配图已生成：fig_goals_poisson / fig_goaldiff_normal / fig_team_beta / fig_goalminute_uniform")
