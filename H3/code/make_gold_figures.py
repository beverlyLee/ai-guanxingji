# -*- coding: utf-8 -*-
"""
AI观星记 · H3（高数·黄金）科学风配图脚本
从 CSV 重生 4 张图，零编造、可复现。
数据：上金所 Au99.99 逐日收盘（7/21–8/24，25 交易日） + 长周期年均锚点。
风格：STHeiti 中文 / Okabe-Ito 色盲安全 / 去脊线 / dpi150+300 / 无 chart junk。
"""
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---------- 字体 ----------
FONT = "Heiti SC"
if not any(f.name == FONT for f in fm.fontManager.ttflist):
    FONT = "PingFang SC"
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

# ---------- Okabe-Ito 色盲安全配色 ----------
C = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}

BASE = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_H3_高数_黄金"

def load_daily():
    dates, closes = [], []
    with open(f"{BASE}/data/au99_daily_2026.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))
            closes.append(float(row["close_yuan_per_g"]))
    return np.array(dates), np.array(closes, float)

def load_annual():
    yrs, avgs, notes = [], [], []
    with open(f"{BASE}/data/au_annual_avg.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            yrs.append(int(row["year"])); avgs.append(float(row["avg_yuan_per_g"]))
            notes.append(row["note"])
    return np.array(yrs), np.array(avgs, float), notes

def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(length=0)

dates, P = load_daily()
yrs, avgs, _ = load_annual()

# =====================================================================
# fig1：金价破千全景（上：长周期年均；下：逐日 7/21–8/24 标出千元线）
# =====================================================================
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), dpi=150)
fig.subplots_adjust(hspace=0.32, left=0.10, right=0.95, top=0.93, bottom=0.07)

# 上：长周期年均
a1.plot(yrs, avgs, "-o", color=C["verm"], lw=2.4, ms=7, zorder=3)
for x, y in zip(yrs, avgs):
    a1.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, color=C["black"])
a1.axhline(1000, color=C["purple"], ls="--", lw=1.4)
a1.annotate("1000 元/克", (yrs[-1], 1000), color=C["purple"], fontsize=9,
             xytext=(-4, -12), textcoords="offset points", ha="right")
a1.set_title("上金所 Au99.99 年均价：四年爬坡，2026 破千", fontsize=13, weight="bold")
a1.set_ylabel("元/克")
a1.set_xticks(yrs)
a1.set_ylim(300, 1120)
style(a1)

# 下：逐日
a2.plot(dates, P, "-o", color=C["blue"], lw=2.2, ms=4, zorder=3, label="Au99.99 收盘")
a2.axhline(1000, color=C["purple"], ls="--", lw=1.4, label="1000 元/克")
i_break = int(np.argmax(P >= 1000))
a2.scatter([dates[i_break]], [P[i_break]], color=C["verm"], s=90, zorder=5)
a2.annotate(f"8/24 收盘 {P[i_break]:.2f}\n国内首次破千", (dates[i_break], P[i_break]),
             xytext=(-10, -42), textcoords="offset points", ha="right", fontsize=9.5,
             color=C["verm"], arrowprops=dict(arrowstyle="-", color=C["verm"], lw=1))
a2.set_title("2026 年 7/21–8/24 逐日收盘：从 887 冲到 1004", fontsize=13, weight="bold")
a2.set_ylabel("元/克"); a2.set_ylim(860, 1030)
a2.tick_params(axis="x", rotation=0)
import matplotlib.dates as mdates
a2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
a2.legend(frameon=False, loc="lower right")
style(a2)
fig.savefig(f"{BASE}/figures/fig1_gold_break1000.png", dpi=300)
print("fig1 done")

# =====================================================================
# fig2：日收益 → 累计（牛顿莱布尼茨离散版 / 对数收益）
# =====================================================================
simp_ret = P[1:] / P[:-1] - 1.0                 # 简单日收益率
log_ret = np.log(P[1:] / P[:-1])               # 对数日收益率
cum_simple = np.cumsum(simp_ret)               # 简单相加（会高估）
cum_log = np.cumsum(log_ret)                   # 对数相加 = 真实累计对数收益
true_cum = P[-1] / P[0] - 1.0                   # 真实累计涨幅
print(f"真实累计涨幅 = {true_cum*100:.2f}%  简单相加={cum_simple[-1]*100:.2f}%  "
      f"对数求和转回={ (np.exp(cum_log[-1])-1)*100:.2f}%")

fig, (b1, b2) = plt.subplots(2, 1, figsize=(8, 7), dpi=150)
fig.subplots_adjust(hspace=0.32, left=0.10, right=0.95, top=0.93, bottom=0.07)
xd = dates[1:]
b1.bar(xd, simp_ret * 100, color=C["sky"], width=0.6, zorder=2)
b1.axhline(0, color=C["black"], lw=0.8)
b1.set_title("每天涨多少？日收益率（变化率的快照）", fontsize=13, weight="bold")
b1.set_ylabel("日收益率 %")
b1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
style(b1)

b2.plot(xd, cum_simple * 100, "-o", color=C["orange"], lw=2, ms=3,
        label=f"简单日收益相加 = {cum_simple[-1]*100:.2f}%")
b2.plot(xd, (np.exp(np.cumsum(log_ret)) - 1) * 100, "-o", color=C["green"], lw=2.2, ms=3,
        label=f"对数收益求和转回 = {(np.exp(cum_log[-1])-1)*100:.2f}%")
b2.axhline(true_cum * 100, color=C["purple"], ls="--", lw=1.4,
           label=f"真实累计 = {true_cum*100:.2f}%")
b2.set_title("攒起来涨多少？简单相加是近似，对数收益求和才精确", fontsize=13, weight="bold")
b2.set_ylabel("累计涨幅 %"); b2.legend(frameon=False, fontsize=9.5)
b2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
style(b2)
fig.savefig(f"{BASE}/figures/fig2_newton_leibniz_cum.png", dpi=300)
print("fig2 done")

# =====================================================================
# fig3：泰勒逐阶逼近（在 8/12 处展开，向左右外推 ±2 天）
# 说明：n 阶项分母有个 n!（0!=1, 1!=1, 2!=2）逐项出现；
#       阶越高、展开点附近贴合越好，远处仍会偏（这是泰勒的本性）。
# =====================================================================
t = np.arange(len(P))
t0 = int(np.where((dates == datetime(2026, 8, 12)))[0][0])
P0 = P[t0]
P1 = (P[t0 + 1] - P[t0 - 1]) / 2.0
P2 = (P[t0 + 1] - 2 * P[t0] + P[t0 - 1])
print(f"t0=8/12  P0={P0:.2f}  P'={P1:.3f}  P''={P2:.3f}")

# h ∈ [-2, +2]，5 个点，看清"展开点附近谁更准"
left, right = 2, 2
tt = np.arange(t0 - left, t0 + right + 1)
PP = P[tt]
hh = tt - t0
T0 = np.full_like(hh, P0, float)
T1 = P0 + P1 * hh
T2 = P0 + P1 * hh + P2 * hh**2 / 2.0

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
fig.subplots_adjust(left=0.11, right=0.95, top=0.91, bottom=0.11)
ax.plot(tt, PP, "-o", color=C["black"], lw=2.6, ms=6, zorder=5, label="实际金价")
ax.plot(tt, T0, color=C["orange"], lw=1.8, ls=":", label="0 阶（常数，÷0!）")
ax.plot(tt, T1, color=C["sky"], lw=2.0, ls="--", label="1 阶（线性，÷1!）")
ax.plot(tt, T2, color=C["green"], lw=2.0, ls="-.", label="2 阶（含曲率，÷2!）")
ax.axvline(t0, color=C["verm"], ls=":", lw=1.2)
ax.annotate("展开点 8/12", (t0, P0), color=C["verm"], fontsize=10,
             xytext=(8, 14), textcoords="offset points")
ax.annotate("线性外推走偏", (t0 + 1, T1[left + 1]),
             xytext=(8, 14), textcoords="offset points", fontsize=9, color=C["sky"],
             arrowprops=dict(arrowstyle="-", color=C["sky"], lw=0.8))
ax.annotate("二阶在展开点\n附近贴合", (t0 + 1, T2[left + 1] - 0.5),
             xytext=(8, -42), textcoords="offset points", fontsize=9, color=C["green"],
             arrowprops=dict(arrowstyle="-", color=C["green"], lw=0.8))
ax.set_title("只知道当下金价+斜率，怎样预判明天？泰勒展开逐阶逼近",
             fontsize=12.5, weight="bold")
ax.set_ylabel("元/克")
labels = [dates[int(i)].strftime("%m/%d") for i in tt]
ax.set_xticks(tt); ax.set_xticklabels(labels)
ax.set_ylim(910, 970)
ax.legend(frameon=False, fontsize=9.5, loc="lower center")
style(ax)
fig.savefig(f"{BASE}/figures/fig3_taylor_orders.png", dpi=300)
print("fig3 done")

# =====================================================================
# fig4：两资产风险-收益前沿（拉格朗日求最小方差组合）
# =====================================================================
# 黄金：基于 7/21–8/24 窗口的真实测算（年化，仅示例）
mu_g = float(np.mean(log_ret) * 252)
sig_g = float(np.std(log_ret, ddof=1) * np.sqrt(252))
# 股票指数：示例参数（明确标注，非真实推荐）
mu_s, sig_s, rho = 0.08, 0.18, 0.10
print(f"黄金(窗口测算) 年化收益≈{mu_g*100:.1f}%  年化波动≈{sig_g*100:.1f}%")
w = np.linspace(0, 1, 201)
sig_p = np.sqrt(w**2 * sig_g**2 + (1 - w)**2 * sig_s**2 + 2 * w * (1 - w) * rho * sig_g * sig_s)
mu_p = w * mu_g + (1 - w) * mu_s
# 全局最小方差组合（拉格朗日闭式）
w_min = (sig_s**2 - rho * sig_g * sig_s) / (sig_g**2 + sig_s**2 - 2 * rho * sig_g * sig_s)
i_min = int(np.argmin(sig_p))
print(f"最小方差组合 黄金权重 w* = {w_min:.3f}")

fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.11)
ax.plot(sig_p * 100, mu_p * 100, color=C["blue"], lw=2.4, zorder=3, label="两资产有效前沿")
ax.scatter([sig_g * 100], [mu_g * 100], color=C["verm"], s=80, zorder=5)
ax.annotate("黄金", (sig_g * 100, mu_g * 100), xytext=(8, -4),
             textcoords="offset points", fontsize=10, color=C["verm"])
ax.scatter([sig_s * 100], [mu_s * 100], color=C["orange"], s=80, zorder=5)
ax.annotate("股票指数（示例）", (sig_s * 100, mu_s * 100), xytext=(-8, 10),
             textcoords="offset points", fontsize=10, color=C["orange"], ha="right")
ax.scatter([sig_p[i_min] * 100], [mu_p[i_min] * 100], color=C["green"], s=140, zorder=6,
            marker="*")
ax.annotate(f"最小方差组合\n黄金 {w_min*100:.0f}% / 股票 { (1-w_min)*100:.0f}%",
             (sig_p[i_min] * 100, mu_p[i_min] * 100), xytext=(14, -8),
             textcoords="offset points", ha="left", fontsize=9.5, color=C["green"])
ax.set_title("想赚 X%，怎样把波动压到最小？带约束求最优（拉格朗日）", fontsize=12.5, weight="bold")
ax.set_xlabel("年化波动（风险） %"); ax.set_ylabel("年化收益（目标） %")
ax.legend(frameon=False, loc="lower center", fontsize=9.5, bbox_to_anchor=(0.5, 0.02))
style(ax)
fig.savefig(f"{BASE}/figures/fig4_lagrange_frontier.png", dpi=300)
print("fig3 done" if False else "fig4 done")
print("ALL FIGURES SAVED")
