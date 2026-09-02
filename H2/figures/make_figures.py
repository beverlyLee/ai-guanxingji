#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2 配图生成 —— scientific-visualization 风格
STHeiti 中文 / Okabe-Ito 色盲安全 / dpi150+300 / 去脊线 / 无 chart junk
读取 results.json + CSV，输出 5 张 PNG 到 figures/
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- 字体：STHeiti（系统中注册名为 Heiti SC） ----
plt.rcParams["font.family"] = "Heiti SC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# ---- Okabe-Ito 色盲安全配色 ----
C = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "verm": "#D55E00", "purple": "#CC79A7",
}

FIG = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_H2_积分_NEV/figures"
DATA = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_H2_积分_NEV/data"
R = json.load(open(f"{DATA}/results.json", encoding="utf-8"))
T = np.array(R["T"]); V = np.array(R["V"])
cum_rect = np.array(R["cum_rect"]); cum_trap = np.array(R["cum_trap"])
years = R["years"]; official = R["official"]

def year_label(t):
    y = 2018 + int(round(t)) // 12
    return y

# =========================================================
# 图① 月度销量曲线 + 下方面积（积分=累计）双轴
# =========================================================
fig, ax1 = plt.subplots(figsize=(9, 4.6))
x = T
ax1.fill_between(x, V, color=C["sky"], alpha=0.28, label="月销量（速率）下方面积 = 积分")
ax1.plot(x, V, color=C["blue"], lw=1.8, zorder=3, label="月度零售（万辆/月）")
# 2020 缺口标注
ax1.axvspan(24 - 0.5, 36 - 0.5, color=C["black"], alpha=0.05)
ax1.text(30, max(V) * 0.9, "2020 缺（仅官方年度锚点）", ha="center", va="top",
         fontsize=8.5, color=C["black"], alpha=0.6)
ax1.set_ylabel("月度零售（万辆 / 月）", color=C["blue"])
ax1.tick_params(axis="y", labelcolor=C["blue"])
ax1.set_xlabel("时间（2018-01 起，单位：月）")
ax1.set_xticks([0, 12, 24, 36, 48, 60, 72, 84, 95])
ax1.set_xticklabels(["18-01", "19-01", "20-01", "21-01", "22-01", "23-01", "24-01", "25-01", "25-12"])
ax2 = ax1.twinx()
ax2.plot(x, cum_rect, color=C["verm"], lw=2.4, zorder=4, label="累计（逐月求和 = 积分）")
ax2.set_ylabel("累计零售（万辆）", color=C["verm"])
ax2.tick_params(axis="y", labelcolor=C["verm"])
# 合并图例
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, frameon=False)
for s in ["top", "right"]:
    ax1.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
ax1.grid(axis="y", ls=":", alpha=0.35)
fig.suptitle("月销量是速率，把它“加起来”就是累计 —— 这就是积分", fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_monthly_cumulative.png"); plt.close(fig)

# =========================================================
# 图② 逐年内：我的求和 vs 乘联会官方年度零售（误差对照）
# =========================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
yrs = [y for y in years]
my = [R["year_err"][str(y)]["sum"] for y in yrs]
off = [official[str(y)] for y in yrs]
xpos = np.arange(len(yrs))
w = 0.38
b1 = ax.bar(xpos - w/2, my, w, color=C["blue"], label="本文求和（积分）")
b2 = ax.bar(xpos + w/2, off, w, color=C["orange"], label="乘联会官方年度零售")
for i, y in enumerate(yrs):
    err = R["year_err"][str(y)]["err_sum"]
    ax.text(xpos[i], max(my[i], off[i]) + 12, f"{err:+.1f}%", ha="center",
            fontsize=8.5, color=C["verm"])
ax.set_xticks(xpos); ax.set_xticklabels(yrs)
ax.set_ylabel("年度零售（万辆）")
ax.set_xlabel("年份（2020 无逐月数据，未参与）")
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, max(off) * 1.18)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.35)
fig.suptitle("把 12 个月“积分”一遍，和官方年度数差多少？", fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_year_error.png"); plt.close(fig)

# =========================================================
# 图③ 解析积分 vs 数值积分（累计曲线三条）
# =========================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(x, cum_rect, color=C["verm"], lw=2.6, zorder=4, label="数值积分（逐月求和）· 基准")
# 多项式解析积分曲线：用 deg=2 闭式原函数在每月端点求值
# np.polyfit/np.polyval 一致：系数按降幂（高次在前）
pbest = np.array(R["poly_res"][str(R["best_deg"])]["poly"])  # 降幂 [c2, c1, c0]
intc = [c / (len(pbest) - 1 - i + 1) for i, c in enumerate(pbest)] + [0.0]  # 降幂原函数
intpoly = np.array(intc)
cum_poly = np.polyval(intpoly, T) - np.polyval(intpoly, T[0])
ax.plot(x, cum_poly, color=C["green"], lw=2.0, ls="--", zorder=3,
        label=f"解析积分（多项式 deg={R['best_deg']}）")
ax.plot(x, cum_trap, color=C["orange"], lw=1.4, ls=":", zorder=2,
        label="梯形数值积分（当瞬时采样）")
ax.set_ylabel("累计零售（万辆）")
ax.set_xlabel("时间（月，2018-01 起）")
ax.set_xticks([0, 12, 24, 36, 48, 60, 72, 84, 95])
ax.set_xticklabels(["18-01", "19-01", "20-01", "21-01", "22-01", "23-01", "24-01", "25-01", "25-12"])
ax.legend(frameon=False, fontsize=9, loc="upper left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.35)
fig.suptitle("三种积分，三种答案：光滑模型把季节低谷填平了", fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_analytic_vs_numeric.png"); plt.close(fig)

# =========================================================
# 图④ 导数/积分 对偶示意（H1 vs H2）
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
# 左：温度曲线 + 切线（求导）
ax = axes[0]
xt = np.linspace(0, 10, 200)
yt = 0.04 * xt**2 + 0.1 * xt + 2.0
ax.plot(xt, yt, color=C["verm"], lw=2.2)
x0 = 6.0
y0 = 0.04 * x0**2 + 0.1 * x0 + 2.0
sl = 0.08 * x0 + 0.1
xs = np.array([x0 - 2, x0 + 2])
ax.plot(xs, y0 + sl * (xs - x0), color=C["black"], lw=1.6, ls="--")
ax.scatter([x0], [y0], color=C["verm"], zorder=5, s=28)
ax.annotate("切线斜率 = 升温速率\n（求导：曲线→斜率）", xy=(x0, y0), xytext=(x0 - 4.2, y0 + 1.2),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C["black"]))
ax.set_title("H1 · 求导", fontsize=11)
ax.set_xlabel("时间"); ax.set_ylabel("全球温度 anomaly")
# 右：月销量柱 + 下方面积（积分）
ax = axes[1]
xr = np.arange(1, 13)
vr = np.array([40, 27, 44, 38, 52, 60, 58, 66, 72, 70, 78, 90])
ax.bar(xr, vr, color=C["sky"], alpha=0.85, width=0.7)
ax.fill_between(xr, vr, color=C["blue"], alpha=0.30, step="mid")
ax.annotate("下方面积 = 全年累计\n（积分：速率→总量）", xy=(6.5, 60), xytext=(2.4, 78),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C["black"]))
ax.set_title("H2 · 积分", fontsize=11)
ax.set_xlabel("月份"); ax.set_ylabel("月度零售（万辆）")
for ax in axes:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.35)
fig.suptitle("同一套工具，正反两面用：H1 求导看“变多快”，H2 积分看“一共多少”", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_duality.png"); plt.close(fig)

# =========================================================
# 图⑤ 年度零售增长（8 年涨 ~13 倍）+ 2025 渗透率破 50%
# =========================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
allyrs = list(range(2018, 2026))
vals = [official[str(y)] for y in allyrs]
colors = [C["blue"] if y != 2025 else C["verm"] for y in allyrs]
bars = ax.bar(allyrs, vals, color=colors, width=0.66)
for y, v in zip(allyrs, vals):
    ax.text(y, v + 18, f"{v:.0f}", ha="center", fontsize=8.5, color=C["black"])
ax.set_ylabel("年度零售（万辆）")
ax.set_xlabel("年份")
ax.set_ylim(0, max(vals) * 1.18)
ax.annotate("2025 渗透率首破 50%\n全年 1280.9 万，较 2018 涨 ~13 倍",
            xy=(2025, vals[-1]), xytext=(2021.2, vals[-1] * 0.62),
            fontsize=9, color=C["verm"],
            arrowprops=dict(arrowstyle="->", color=C["verm"]))
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.35)
fig.suptitle("新能源乘用车：8 年涨 13 倍，2025 渗透率破 50%", fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_annual_growth.png"); plt.close(fig)

print("✅ 5 张图已生成：")
import os
for f in sorted(os.listdir(FIG)):
    if f.endswith(".png"):
        print("  ", f, os.path.getsize(f"{FIG}/{f}"), "bytes")
