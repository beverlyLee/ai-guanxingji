"""
M9 配图：全部由 stats.json / phone_clustered.csv 真实重绘，不依赖任何外部数据。

风格：STHeiti 中文、Okabe-Ito 色盲安全配色、去脊线、无 chart junk、dpi 150/300

运行：python code/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
if not FIG.is_dir():
    FIG.mkdir(parents=True)

plt.rcParams.update({
    "font.sans-serif": ["STHeiti", "Heiti TC", "PingFang SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4D4D4D",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.frameon": False,
})
BLUE, VERM, GREEN, PINK, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"

S = json.loads((ROOT / "stats.json").read_text(encoding="utf-8"))
d = pd.read_csv(ROOT / "data" / "phone_clustered.csv")
Z = np.load(ROOT / "data" / "Z.npy")
PC = np.load(ROOT / "data" / "PC.npy")
lab_km = np.load(ROOT / "data" / "labels_km.npy")
lab_db = np.load(ROOT / "data" / "labels_db.npy")

CN = {"price_eur": "价格(欧元)", "screen_in": "屏幕(英寸)", "weight_g": "重量(克)",
      "battery_mah": "电池(mAh)", "ppi": "像素密度(ppi)", "storage_gb": "存储(GB)",
      "ram_gb": "内存(GB)", "camera_mp": "主摄(MP)"}


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print(f"已写出 {p}")


# ---------- fig1：为什么必须先取对数 ----------
sk_raw = S["skew_raw"]
sk_log = S["skew_after_log"]
LOGGED = set(S["logged_features"])          # 真正做了 log1p 的特征
order = sorted(sk_raw, key=lambda k: sk_raw[k])
y = np.arange(len(order))
h = 0.38
fig, ax = plt.subplots(figsize=(8.8, 5.4))
for i, k in enumerate(order):
    # 取过对数的用实色，没取对数的（偏度本就小）用浅灰，避免"图上 8 个、标题说 5 个"的歧义
    c_raw = VERM if k in LOGGED else "#C4C4C4"
    c_log = BLUE if k in LOGGED else "#9E9E9E"
    ax.barh(i + h / 2, sk_raw[k], height=h, color=c_raw)
    ax.barh(i - h / 2, sk_log[k], height=h, color=c_log)
ax.axvline(0, color="#A6A6A6", lw=0.8)
ax.axvline(1.0, color="#8C8C8C", lw=1, ls="--")
ax.text(1.06, -0.55, "偏度 = 1\n超过就右偏严重", fontsize=9, color="#595959", va="bottom")
ax.set_yticks(y)
ax.set_yticklabels([CN[k] for k in order])
ax.set_xlabel("偏度（数值越大，长尾越长）")
ax.set_title("五个重右偏特征取对数后，偏度从 1.0–3.1 压到 −0.2–0.7")
ax.legend(handles=[
    Patch(color=VERM, label="原始数值（需取对数的 5 个）"),
    Patch(color=BLUE, label="取对数后（同 5 个）"),
    Patch(color="#C4C4C4", label="未取对数（偏度本就 ≤ 0.8 的 3 个）"),
], loc="lower right", fontsize=9.5)
ax.set_xlim(-0.8, 3.7)
save(fig, "fig1-skew-log.png")

# ---------- fig2：选 K 的两个指标打起来了 ----------
km = S["kmeans"]
ks = km["k_grid"]
fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.plot(ks, km["inertia"], "o-", color=BLUE, lw=2, ms=5, label="簇内平方和（惯性）")
ax.set_xlabel("K（分成几档）")
ax.set_ylabel("惯性（越小越紧凑）", color=BLUE)
ax.tick_params(axis="y", labelcolor=BLUE)

ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(ks, km["silhouette"], "s--", color=VERM, lw=2, ms=5, label="轮廓系数")
ax2.set_ylabel("轮廓系数（越大越分得开）", color=VERM)
ax2.tick_params(axis="y", labelcolor=VERM)

ax.axvline(km["k_by_elbow"], color=BLUE, lw=1.2, ls=":")
ax.annotate(f"肘部法 → K={km['k_by_elbow']}", xy=(km["k_by_elbow"], km["inertia"][km["k_by_elbow"] - 2]),
            xytext=(km["k_by_elbow"] + 0.4, km["inertia"][0] * 0.92), color=BLUE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
ax2.axvline(km["k_by_silhouette"], color=VERM, lw=1.2, ls=":")
ax2.annotate(f"轮廓系数 → K={km['k_by_silhouette']}\n（最高也只有 {km['silhouette_max']}）",
             xy=(km["k_by_silhouette"], km["silhouette"][0]),
             xytext=(km["k_by_silhouette"] + 1.0, 0.20), color=VERM, fontsize=10,
             arrowprops=dict(arrowstyle="->", color=VERM, lw=1))
ax.set_title("两个指标指向两个不同的 K，而且轮廓系数全程不超过 0.37")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)
save(fig, "fig2-k-select.png")

# ---------- fig3：gap 统计量对参照系极其敏感 ----------
gs = S["gap_statistic"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
for ax, names, cols, title in [
    (axes[0], ["uniform_box", "uniform_pca"], [SKY, BLUE],
     "参照 = 均匀分布（逐维 / 沿主轴）"),
    (axes[1], ["gaussian"], [VERM],
     "参照 = 多元高斯（保住均值与协方差）"),
]:
    for name, color in zip(names, cols):
        tab = gs[name]["table"]
        kk = sorted(int(k) for k in tab)
        g = [tab[str(k)]["gap"] for k in kk]
        sd = [tab[str(k)]["sd"] for k in kk]
        label = {"uniform_box": "逐维均匀", "uniform_pca": "沿主成分均匀",
                 "gaussian": "多元高斯"}[name]
        ax.errorbar(kk, g, yerr=sd, fmt="o-", color=color, lw=2, ms=5,
                    capsize=3, label=label)
    ax.set_xlabel("K")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
axes[0].set_ylabel("gap 统计量")
axes[0].text(0.03, 0.96, "两条线都一路上涨、找不到拐点\n→ 这种参照下选不出 K",
             transform=axes[0].transAxes, fontsize=10, color="#4D4D4D",
             ha="left", va="top")
axes[0].set_ylim(0.70, 1.22)
axes[1].annotate(f"拐点在 K={gs['gaussian']['k_selected']}",
                 xy=(4, gs["gaussian"]["table"]["4"]["gap"]),
                 xytext=(4.7, gs["gaussian"]["table"]["1"]["gap"] * 3),
                 color=VERM, fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=1))
fig.suptitle("同一批数据、同一个公式，只换参照系：结论就从「选不出」变成「K=4」", fontsize=13, y=1.02)
save(fig, "fig3-gap-references.png")

# ---------- fig4：PCA 平面上的四档 + 它们价格重叠得厉害 ----------
CL_NAME = {3: "小屏入门", 2: "大屏低价", 1: "小屏高清", 0: "旗舰堆料"}
cl_order = [3, 2, 1, 0]
colors = {3: BLUE, 2: GREEN, 1: ORANGE, 0: VERM}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
for c in cl_order:
    m = lab_km == c
    ax.scatter(PC[m, 0], PC[m, 1], s=9, alpha=0.55, color=colors[c],
               label=f"{CL_NAME[c]}（n={int(m.sum())}）", linewidths=0)
ax.set_xlabel(f"PC1（解释 {S['pca_explained_variance_ratio'][0]*100:.1f}% 方差）")
ax.set_ylabel(f"PC2（解释 {S['pca_explained_variance_ratio'][1]*100:.1f}%）")
ax.set_title("K=4 的四档在平面上的样子")
ax.legend(fontsize=9, markerscale=2.2, loc="upper left")

ax = axes[1]
data = [d.loc[lab_km == c, "price_eur"].values for c in cl_order]
bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                medianprops=dict(color="#333333", lw=1.6))
for patch, c in zip(bp["boxes"], cl_order):
    patch.set_facecolor(colors[c])
    patch.set_alpha(0.55)
    patch.set_edgecolor("#4D4D4D")
ax.set_yticklabels([f"{CL_NAME[c]}\n中位 €{np.median(d.loc[lab_km == c, 'price_eur']):.0f}"
                    for c in cl_order], fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("价格（欧元，对数刻度）")
ax.set_title("四档的价格区间彼此大幅重叠")
ov = S["price_band_overlap_width"]
ax.text(0.98, 0.05, f"相邻档重叠宽度：{ov[0]:.0f} / {ov[1]:.0f} / {ov[2]:.0f} 欧元",
        transform=ax.transAxes, ha="right", fontsize=10, color=VERM)
save(fig, "fig4-pca-clusters-price.png")

# ---------- fig5：DBSCAN 的噪声与富集检验 ----------
db = S["dbscan"]
en = db["enrichment"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
norm = lab_db != -1
ax.scatter(PC[norm, 0], PC[norm, 1], s=7, color="#C9C9C9", alpha=0.6,
           linewidths=0, label=f"被判进簇的 {int(norm.sum())} 款")
noise = lab_db == -1
fold = (d["is_fold"] == 1).values & noise
rug = (d["is_rugged"] == 1).values & noise
oth = noise & ~fold & ~rug
ax.scatter(PC[oth, 0], PC[oth, 1], s=16, color=BLUE, alpha=0.75, linewidths=0,
           label=f"噪声（其他）{int(oth.sum())} 款")
ax.scatter(PC[fold, 0], PC[fold, 1], s=64, marker="^", color=VERM, edgecolors="white",
           linewidths=0.8, label=f"折叠屏 {int(fold.sum())} 款")
ax.scatter(PC[rug, 0], PC[rug, 1], s=48, marker="s", color=GREEN, edgecolors="white",
           linewidths=0.8, label=f"三防巨电池 {int(rug.sum())} 款")
ax.set_xlabel(f"PC1（{S['pca_explained_variance_ratio'][0]*100:.1f}%）")
ax.set_ylabel(f"PC2（{S['pca_explained_variance_ratio'][1]*100:.1f}%）")
ax.set_title(f"DBSCAN 标出的噪声里，异类机型扎堆")
ax.legend(fontsize=9, loc="upper left")

ax = axes[1]
bars = [en["expected"], en["observed"]]
ax.bar([0, 1], bars, width=0.55, color=["#B8B8B8", VERM])
ax.set_xticks([0, 1])
ax.set_xticklabels([f"随机抽\n{en['noise_n']} 款\n的期望", f"DBSCAN\n实际标出"], fontsize=10)
for i, v in enumerate(bars):
    ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=12, color="#333333")
ax.set_ylabel("噪声中「折叠屏 + 三防巨电池」的款数")
ax.set_title(f"富集 {en['enrichment']} 倍，超几何检验 p = {en['p_value']:.1e}")
ax.set_ylim(0, max(bars) * 1.28)
save(fig, "fig5-dbscan-noise.png")

print("\n5 张图全部由 stats.json 与 phone_clustered.csv 重绘完成。")
