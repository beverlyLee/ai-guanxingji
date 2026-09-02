#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 新数据集（NASA GISTEMP 1880-2025）配图 —— 出版级风格
配色 Okabe-Ito 色盲安全；字体 STHeiti；去脊线；dpi 150/300。
所有拟合均从 gistemp_1880_2025.csv 重新推导，保证可复现。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["STHeiti", "Songti SC", "SimHei", "Arial"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

BASE = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/assets/h1-new-data"
OUT = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/assets/h1-gistemp-figures"
os.makedirs(OUT, exist_ok=True)
CSV = f"{BASE}/gistemp_1880_2025.csv"

# ---------- 读取 ----------
years, vals = [], []
with open(CSV) as f:
    f.readline()
    for line in f:
        line = line.strip()
        if not line:
            continue
        yr, v = line.split(",")
        years.append(int(yr)); vals.append(float(v))
years = np.array(years, float)
y = np.array(vals, float)
x = years - 1880.0
t = x / 10.0
n = len(x)

# ---------- 拟合（闭式） ----------
# 直线
A = np.vstack([x, np.ones_like(x)]).T
w, b = np.linalg.lstsq(A, y, rcond=None)[0]
# 二次
A2 = np.vstack([x**2, x, np.ones_like(x)]).T
a2, b2, c2 = np.linalg.lstsq(A2, y, rcond=None)[0]
# 指数（LM，与 verify 一致）
def lm_exp(xd, yd, p0=[0.001, 0.049]):
    p = np.array(p0, float); lam = 1e-3
    r = p[0]*np.exp(p[1]*xd) - yd; cost = np.dot(r, r)
    for _ in range(2000):
        J = np.empty((len(xd), 2))
        J[:, 0] = np.exp(p[1]*xd); J[:, 1] = p[0]*xd*np.exp(p[1]*xd)
        H = J.T @ J; g = J.T @ r; d = np.diag(H).copy(); d[d==0]=1e-12
        dp = np.linalg.solve(H + lam*np.diag(d), -g)
        pn = p + dp; rn = pn[0]*np.exp(pn[1]*xd) - yd; cn = np.dot(rn, rn)
        if cn < cost: p, r, cost, lam = pn, rn, cn, max(lam/3, 1e-8)
        else: lam = min(lam*3, 1e6)
        if np.max(np.abs(dp)) < 1e-10: break
    return p
ae, ce = lm_exp(x, y)
print("拟合系数 -> 直线 w,b:", round(w,5), round(b,5), "| 二次 a,b,c:", round(a2,8), round(b2,5), round(c2,5), "| 指数 a,c:", round(ae,6), round(ce,5))

def r2(poly, xd, yd):
    return 1 - np.sum((yd-np.polyval(poly,xd))**2)/np.sum((yd-np.mean(yd))**2)
r2_lin = r2(np.array([w,b]), x, y)
r2_q = r2(np.array([a2,b2,c2]), x, y)
yhat_e = ae*np.exp(ce*x)
r2_e = 1 - np.sum((y-yhat_e)**2)/np.sum((y-np.mean(y))**2)
rms = lambda yh: np.sqrt(np.mean((y-yh)**2))
print("R² -> 直线", round(r2_lin,4), "二次", round(r2_q,4), "指数", round(r2_e,4))
print("RMS -> 直线", round(rms(w*x+b),4), "二次", round(rms(np.polyval([a2,b2,c2],x)),4), "指数", round(rms(yhat_e),4))

# =====================================================================
# 图1：学习率对比（凸损失上的下降轨迹）—— Q3 梯度下降引子
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
xx = np.linspace(-3, 3, 400)
L = 0.5*(xx-0.2)**2 + 0.3
ax.plot(xx, L, color=OKABE[6], lw=2.4, label="损失函数 L(θ)")
ax.set_xlabel("参数 θ"); ax.set_ylabel("损失 L(θ)")
ax.set_title("学习率 η：决定每次下坡迈多大一步")
def descent(eta, x0, steps=14):
    pts=[x0]; xc=x0
    for _ in range(steps):
        xc = xc - eta*(xc-0.2); pts.append(xc)
    return np.array(pts)
for eta, c, lab in [(0.05, OKABE[2], "η 太小：步子碎，慢悠悠"),
                    (0.45, OKABE[0], "η 合适：平稳滑到谷底"),
                    (0.95, OKABE[1], "η 太大：冲过头，来回蹦")]:
    pts = descent(eta, -2.6); ys = 0.5*(pts-0.2)**2+0.3
    ax.plot(pts, ys, "o-", color=c, ms=5, lw=1.6, alpha=0.9, label=lab)
ax.axvline(0.2, color="grey", ls="--", lw=1, alpha=0.6)
ax.text(0.25, 0.35, "谷底 θ*", color="grey", fontsize=11)
ax.legend(frameon=False, loc="upper right", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT, "fig1-lr-compare.png")); plt.close(fig)

# =====================================================================
# 图2：原始数据 + 三模型曲线叠加 —— 一图看清谁更贴
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(years, y, s=14, color=OKABE[6], alpha=0.55, label="观测值（146 年）", zorder=3)
xs = np.linspace(x.min(), x.max(), 300)
ax.plot(years, np.polyval([w,b], x), color=OKABE[0], lw=2.2, label=f"直线 R²={r2_lin:.3f}")
ax.plot(years, np.polyval([a2,b2,c2], x), color=OKABE[1], lw=2.2, label=f"二次 R²={r2_q:.3f}")
ax.plot(years, ae*np.exp(ce*x), color=OKABE[2], lw=2.2, label=f"指数 R²={r2_e:.3f}")
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.set_xlabel("年份"); ax.set_ylabel("温度异常 (°C，相对 1951–1980)")
ax.set_title("1880–2025 全球地表温度异常：直线/二次/指数 三模型对比")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT, "fig2-data-models.png")); plt.close(fig)

# =====================================================================
# 图3：残差 RMS 对比（模型选择硬证据）
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
names = ["直线", "二次", "指数"]
rmsv = [rms(w*x+b), rms(np.polyval([a2,b2,c2],x)), rms(yhat_e)]
cols = [OKABE[0], OKABE[1], OKABE[2]]
bars = ax.bar(names, rmsv, color=cols, width=0.55)
for bar, v in zip(bars, rmsv):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=12)
ax.set_ylabel("残差 RMS (°C)")
ax.set_title("谁的残差更小？RMS 越低 = 拟合越贴")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT, "fig3-residuals.png")); plt.close(fig)

# =====================================================================
# 图4：梯度下降收敛 —— 特征缩放前后（真实教学点）
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 左：不缩放 x=0..145, eta=0.01 -> 发散
ws, bs, loss = [], [], []
wc, bc = 0.0, 0.0; eta=0.01
for i in range(2000):
    yh = wc*x + bc; e = yh - y
    dw = (2/n)*np.sum(e*x); db = (2/n)*np.sum(e)
    wc -= eta*dw; bc -= eta*db
    cur_loss = np.mean(e**2)
    if not np.isfinite(wc) or abs(wc) > 1e4 or cur_loss > 1e6:
        break
    ws.append(wc); bs.append(bc); loss.append(cur_loss)
axes[0].plot(range(len(loss)), loss, color=OKABE[1], lw=2)
axes[0].set_yscale("log")
axes[0].set_xlabel("迭代步数"); axes[0].set_ylabel("损失（对数轴）")
axes[0].set_title(f"不缩放 x=yr-1880, η=0.01\n→ 损失面被拉长，{len(loss)} 步内发散")
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
# 右：缩放 t=(yr-1880)/10, eta=0.01 -> 收敛
wt, bt = 0.0, 0.0
wt_hist, loss_t = [], []
for i in range(20000):
    yh = wt*t + bt; e = yh - y
    dw = (2/n)*np.sum(e*t); db = (2/n)*np.sum(e)
    wt -= eta*dw; bt -= eta*db
    if i % 500 == 0 or i == 19999:
        wt_hist.append(wt/10); loss_t.append(np.mean(e**2))
gd_x = list(range(0,20000,500)) + [19999]
axes[1].plot(gd_x, loss_t, color=OKABE[2], lw=2, label="损失")
axes[1].axhline(np.mean((y-(w*x+b))**2), color=OKABE[0], ls="--", lw=1.5, label=f"闭式最优 {np.mean((y-(w*x+b))**2):.4f}")
axes[1].set_xlabel("迭代步数"); axes[1].set_ylabel("损失")
axes[1].set_title(f"缩放 t=(yr-1880)/10, η=0.01\n→ 收敛到 w={wt/10:.5f}, b={bt:.5f}（与闭式一致）")
axes[1].legend(frameon=False, fontsize=10)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4-gd-convergence.png")); plt.close(fig)

# =====================================================================
# 图5：二次导数 = 升温速率曲线 —— 判加速 / 到顶
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
slope = 2*a2*x + b2          # dy/dx
ax.plot(years, slope, color=OKABE[1], lw=2.4)
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.fill_between(years, slope, 0, where=(slope>0), color=OKABE[1], alpha=0.12)
ax.axvline(1909, color=OKABE[3], ls="--", lw=1.2)
ax.text(1909, -0.012, " ≈1909 速率=0", color=OKABE[3], fontsize=10, ha="center")
ax.scatter([2025],[2*a2*145+b2], color=OKABE[0], zorder=5, s=45)
ax.annotate(f"2025: {2*a2*145+b2:.4f} °C/yr", xy=(2025,2*a2*145+b2),
            xytext=(1990,0.030), fontsize=11, color=OKABE[0],
            arrowprops=dict(arrowstyle="->", color=OKABE[0]))
ax.set_xlabel("年份"); ax.set_ylabel("升温速率 dy/dx (°C/yr)")
ax.set_title("二次模型下的升温速率：逐年增大 → 加速升温，未见顶")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT, "fig5-derivative.png")); plt.close(fig)

print("全部 5 张图已生成 ->", OUT)
