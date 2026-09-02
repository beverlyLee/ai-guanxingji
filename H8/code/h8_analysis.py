# -*- coding: utf-8 -*-
"""
H8 分析脚本：概率论（熵 + 回归分析）
数据集：California Housing（公开基准，来源 StatLib / 经 scikit-learn 再分发；
        此处用 A. Géron《Hands-On Machine Learning》仓库的等价 CSV，
        特征构造与 sklearn.fetch_california_housing 完全一致）。
依赖：仅 numpy / scipy / pandas / matplotlib（受管 venv 已装，规避 statsmodels/sklearn）。
"""
import os
import urllib.request
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- 字体（中文无衬线，回退链）----------
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Okabe-Ito 色盲安全配色
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
     "yellow": "#F0E442", "black": "#000000"}

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
CSV_PATH = os.path.join(BASE, "data", "california_housing.csv")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("下载数据 ...")
    urllib.request.urlretrieve(DATA_URL, CSV_PATH)

# ---------- 构造与 sklearn.fetch_california_housing 等价的特征 ----------
df = pd.read_csv(CSV_PATH)
# 公开数据集标准预处理：207 个街区缺失 total_bedrooms，用列中位数填补
n_miss = int(df["total_bedrooms"].isna().sum())
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df["MedInc"] = df["median_income"]                       # 街区收入中位数（万美元）
df["HouseAge"] = df["housing_median_age"]
df["AveRooms"] = df["total_rooms"] / df["households"]     # 户均房间数
df["AveBedrms"] = df["total_bedrooms"] / df["households"] # 户均卧室数
df["Population"] = df["population"]
df["AveOccup"] = df["population"] / df["households"]      # 户均人口
df["Latitude"] = df["latitude"]
df["Longitude"] = df["longitude"]
df["MedHouseVal"] = df["median_house_value"] / 100000.0   # 目标：房价中位数（十万美元）

FEATS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
         "Population", "AveOccup", "Latitude", "Longitude"]
X = df[FEATS].values.astype(float)
y = df["MedHouseVal"].values.astype(float)
N = X.shape[0]
p_intercept = X.shape[1] + 1   # 含截距的参数个数

print("=" * 60)
print(f"样本量 N = {N}（其中 {n_miss} 个街区卧室总数缺失，已用中位数填补）")
print(f"目标 MedHouseVal：均值 = {y.mean():.4f}（十万美元），中位 = {np.median(y):.4f}，"
      f"标准差 = {y.std(ddof=1):.4f}，偏度 = {st.skew(y):.4f}")
print("=" * 60)

# ---------- Q2: 最小二乘 OLS ----------
# 先对 X 做 Z-score 标准化以数值稳定（特征量纲差 4 个数量级会让 SVD 不收敛），
# 再经线性变换 A 把系数换算回原始量纲：β_raw = A⁻¹ g。
Xmean = X.mean(0); Xstd = X.std(0); ystd = y.std(0)
Xz = (X - Xmean) / Xstd
Zc = np.column_stack([np.ones(N), Xz])
g, *_ = np.linalg.lstsq(Zc, y, rcond=None)
yhat = Zc @ g
e = y - yhat
SSR = float(e @ e); SST = float(((y - y.mean()) ** 2).sum())
R2 = 1.0 - SSR / SST
adjR2 = 1.0 - (1.0 - R2) * (N - 1) / (N - p_intercept)
sigma2 = SSR / (N - p_intercept); resid_std = np.sqrt(sigma2)
CovZ = sigma2 * np.linalg.inv(Zc.T @ Zc)
# 原始量纲换算矩阵 A：[1, X] = [1, Xz] @ A
A = np.zeros((p_intercept, p_intercept)); A[0, 0] = 1.0; A[0, 1:] = Xmean
for j in range(1, p_intercept):
    A[j, j] = Xstd[j-1]
Ainv = np.linalg.inv(A)
beta = Ainv @ g
Cov_raw = Ainv @ CovZ @ Ainv.T
se = np.sqrt(np.diag(Cov_raw))
tvals = beta / se
pvals = 2.0 * st.t.sf(np.abs(tvals), N - p_intercept)
F = (R2 / (p_intercept - 1)) / ((1.0 - R2) / (N - p_intercept))
Fp = st.f.sf(F, p_intercept - 1, N - p_intercept)

names = ["截距"] + FEATS
print("\n[Q2] 多元线性回归（最小二乘 OLS）")
print(f"  R²        = {R2:.4f}")
print(f"  调整 R²   = {adjR2:.4f}")
print(f"  残差标准差 = {resid_std:.4f}（十万美元）")
print(f"  F 统计量   = {F:.1f}  (p = {Fp:.2e})")
print("  回归方程系数（单位：房价十万美元 / 自变量单位）：")
for nm, b, s, t, pv in zip(names, beta, se, tvals, pvals):
    star = "***" if pv < 1e-3 else ("**" if pv < 1e-2 else ("*" if pv < 0.05 else ""))
    print(f"    {nm:10s} β={b:+.5f}  se={s:.5f}  t={t:+.1f}  p={pv:.2e} {star}")

# ---------- 标准化系数：量纲拉平，比谁拉动大 ----------
# β*_j = g[j]/ystd（X、y 均 Z-score 时的斜率）
beta_s = np.zeros(p_intercept); beta_s[1:] = g[1:] / ystd
print("\n[Q2] 标准化系数（β* 表示 X 变动 1 个标准差 → y 变动 β* 个标准差）：")
for nm, bs in zip(FEATS, beta_s[1:]):
    print(f"    {nm:10s} β*={bs:+.4f}")

# ---------- Q1: 相关性（Pearson）----------
corr = np.corrcoef(np.column_stack([X, y]).T)
print("\n[Q1] 各特征与房价的 Pearson 相关系数：")
for nm, r in zip(FEATS, corr[:-1, -1]):
    print(f"    {nm:10s} r={r:+.4f}")

# ---------- Q3: 熵 / 互信息（特征重要性，熵口径）----------
def entropy_bins(z, n_bins=15):
    edges = np.linspace(z.min(), z.max(), n_bins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    pk = cnt[cnt > 0] / cnt.sum()
    return float(-np.sum(pk * np.log2(pk)))

HY = entropy_bins(y, 15)
print(f"\n[Q3] 房价 Y 的信息熵 H(Y) = {HY:.4f} bit（15 等宽分箱估计）")

def cond_entropy(y, x, n_bins=15):
    xe = np.linspace(x.min(), x.max(), n_bins + 1)
    xi = np.clip(np.digitize(x, xe) - 1, 0, n_bins - 1)
    H = 0.0
    for b in range(n_bins):
        m = xi == b
        if m.sum() == 0:
            continue
        px = m.sum() / N
        H += px * entropy_bins(y[m], n_bins)
    return H

MI = []
for j in range(X.shape[1]):
    mi = HY - cond_entropy(y, X[:, j])
    MI.append(mi)
MI = np.array(MI)
print("  互信息 I(X;Y)=H(Y)-H(Y|X)（bit）与不确定性解释比例 I/H(Y)：")
mi_order = np.argsort(-MI)
for j in mi_order:
    print(f"    {FEATS[j]:10s} I={MI[j]:.4f}  ({MI[j]/HY*100:.1f}% 的房价不确定性可被该特征单独解释)")

# ---------- 保存数值结果供正文三处一致引用 ----------
res_rows = []
for nm, b, s, t, pv, bs in zip(FEATS, beta[1:], se[1:], tvals[1:], pvals[1:], beta_s[1:]):
    res_rows.append({"feature": nm, "coef": round(b, 5), "se": round(s, 5),
                     "t": round(t, 2), "p": pv, "std_coef": round(bs, 4),
                     "pearson_r": round(corr[:-1, -1][FEATS.index(nm)], 4),
                     "MI": round(MI[FEATS.index(nm)], 4),
                     "MI_frac": round(MI[FEATS.index(nm)] / HY, 4)})
res_df = pd.DataFrame(res_rows)
res_df.to_csv(os.path.join(BASE, "data", "h8_regression_results.csv"), index=False)
print("\n已保存 data/h8_regression_results.csv")

# ================= 绘图 =================
# 图1：房价分布（右偏）
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(y, bins=60, color=C["blue"], alpha=0.85, edgecolor="white", linewidth=0.3)
ax.axvline(y.mean(), color=C["red"], lw=2, label=f"均值 {y.mean():.2f}")
ax.axvline(np.median(y), color=C["orange"], lw=2, ls="--", label=f"中位 {np.median(y):.2f}")
ax.set_xlabel("房价中位数（十万美元）")
ax.set_ylabel("街区数量")
ax.set_title("加州 20640 个街区的房价分布：明显右偏")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig1_price_dist.png")); plt.close(fig)

# 图2：相关热力图
labels = FEATS + ["房价"]
M = corr
fig, ax = plt.subplots(figsize=(7.2, 6))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(M[i, j]) > 0.5 else "black", fontsize=8)
ax.set_title("特征与房价的 Pearson 相关矩阵")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig2_corr_heatmap.png")); plt.close(fig)

# 图3：收入 vs 房价散点 + 单变量 OLS 直线
rng = slice(0, N, 8)
xi = X[rng, 0]; yi = y[rng]
xs = (xi - xi.mean()) / xi.std(); ysm = (yi - yi.mean()) / yi.std()
X1 = np.column_stack([np.ones(len(xi)), xi])
b1, *_ = np.linalg.lstsq(X1, yi, rcond=None)
xs_ord = np.linspace(xi.min(), xi.max(), 100)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(xi, yi, s=6, color=C["blue"], alpha=0.25, label="街区（抽样 1/8）")
ax.plot(xs_ord, b1[0] + b1[1] * xs_ord, color=C["red"], lw=2.5,
        label=f"单变量 OLS：房价 = {b1[0]:.2f} + {b1[1]:.2f}×收入")
ax.set_xlabel("街区收入中位数（万美元）")
ax.set_ylabel("房价中位数（十万美元）")
ax.set_title("收入越高房价越高：但点云在高端被'压平'")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig3_income_fit.png")); plt.close(fig)

# 图4：互信息条形（熵口径特征重要性）
order = mi_order
fig, ax = plt.subplots(figsize=(7, 4.5))
vals = MI[order]
bars = ax.barh([FEATS[o] for o in order][::-1], vals[::-1],
               color=C["green"], alpha=0.85)
ax.set_xlabel("互信息 I(X;Y)（bit）")
ax.set_title("熵口径的特征重要性：谁最能减少房价的不确定性")
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig4_mutual_info.png")); plt.close(fig)

print("已生成 4 张图：figures/fig1..fig4")
print("\n完成。")
