# -*- coding: utf-8 -*-
"""
M12 配图脚本：从 stats.json 与 data/ 重生所有 SVG 风格的科学可视化。
Okabe-Ito 色盲安全配色，去脊线，figure.dpi=150 / savefig.dpi=300，PingFang SC。
运行：python code/figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito 色盲安全配色
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluish": "#0072B2",
    "vermillion": "#D55E00",
    "bluishgreen": "#009E73",
    "reddish": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#999999",
}

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


with open(os.path.join(ROOT, "stats.json"), encoding="utf-8") as f:
    S = json.load(f)

y = np.load(os.path.join(ROOT, "data", "y.npy"))
pca_X = np.load(os.path.join(ROOT, "data", "pca_X.npy"))
pca_sv = np.load(os.path.join(ROOT, "data", "pca_sv.npy"))
xx = np.load(os.path.join(ROOT, "data", "pca_xx.npy"))
yy = np.load(os.path.join(ROOT, "data", "pca_yy.npy"))
Z = np.load(os.path.join(ROOT, "data", "pca_Z.npy"))

c_sweep = S["c_sweep"]
Cv = [r["C"] for r in c_sweep]
margin_v = [r["margin_width"] for r in c_sweep]
nsv_v = [r["n_support"] for r in c_sweep]
nsv_ratio_v = [r["n_support_ratio"] for r in c_sweep]
train_v = [r["train_acc"] for r in c_sweep]
cv_v = [r["cv_acc"] for r in c_sweep]

# ---------- fig1：最大间隔与支持向量（2D PCA 决策边界） ----------
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
            colors=[OI["skyblue"], OI["orange"]], alpha=0.18)
ax.contourf(xx, yy, Z, levels=[-1, 1], colors=[OI["grey"]], alpha=0.18)
ax.contour(xx, yy, Z, levels=[0], colors=[OI["black"]], linewidths=2)
ax.contour(xx, yy, Z, levels=[-1, 1], colors=[OI["black"]], linewidths=1.2, linestyles="--")
ax.scatter(pca_X[y == 0, 0], pca_X[y == 0, 1], s=10, c=OI["skyblue"],
           label="正常邮件", alpha=0.55, edgecolors="none")
ax.scatter(pca_X[y == 1, 0], pca_X[y == 1, 1], s=10, c=OI["vermillion"],
           label="诈骗邮件", alpha=0.55, edgecolors="none")
ax.scatter(pca_X[pca_sv, 0], pca_X[pca_sv, 1], s=46, facecolors="none",
           edgecolors=OI["black"], linewidths=1.3, label="支持向量")
ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2")
ax.set_title("最大间隔分类器：虚线之间是最宽的安全带")
despine(ax); ax.legend(loc="upper right", fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_max_margin.png")); plt.close(fig)

# ---------- fig2：C 与间隔宽度、支持向量数的关系（软间隔权衡） ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
ax1.plot(Cv, margin_v, "-o", color=OI["bluish"], linewidth=2)
ax1.set_xscale("log"); ax1.set_xlabel("惩罚系数 C (log)"); ax1.set_ylabel("间隔宽度 2/‖w‖")
ax1.set_title("C 越小，间隔越宽"); despine(ax1)
ax2.plot(Cv, nsv_v, "-s", color=OI["vermillion"], linewidth=2)
ax2.set_xscale("log"); ax2.set_xlabel("惩罚系数 C (log)"); ax2.set_ylabel("支持向量数量")
ax2.set_title("C 越小，支持向量越多"); despine(ax2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_c_margin.png")); plt.close(fig)

# ---------- fig3：C 与训练/交叉验证准确率 ----------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(Cv, train_v, "-o", color=OI["orange"], linewidth=2, label="训练集准确率")
ax.plot(Cv, cv_v, "-s", color=OI["bluishgreen"], linewidth=2, label="5折交叉验证准确率")
ax.set_xscale("log"); ax.set_xlabel("惩罚系数 C (log)")
ax.set_ylabel("准确率"); ax.set_ylim(0.90, 0.94)
ax.set_title("C 的权衡：太软欠拟合，太硬只是把训练集抠得更紧")
despine(ax); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_c_accuracy.png")); plt.close(fig)

# ---------- fig4：三种核函数 5折CV 准确率与支持向量数 ----------
kc = S["kernel_compare"]
labels = [r["kernel"] for r in kc]
accs = [r["cv_mean_pp"] for r in kc]
nsv = [r["n_support"] for r in kc]
colors = [OI["bluish"], OI["bluishgreen"], OI["vermillion"]]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
bars = ax.bar(labels, accs, color=colors, width=0.55)
for b, a, n in zip(bars, accs, nsv):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.6, f"{a:.1f}%",
            ha="center", fontsize=10, color=OI["black"])
    ax.text(b.get_x() + b.get_width() / 2, 78, f"支持向量 {n}",
            ha="center", fontsize=8, color="white")
ax.set_ylabel("5折交叉验证准确率 (%)"); ax.set_ylim(70, 98)
ax.set_title("核函数对比：RBF 略胜，多项式核在这里翻车")
despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_kernel_compare.png")); plt.close(fig)

# ---------- fig5：诈骗/正常邮件高频词差异（数据从哪来） ----------
tsw = S["top_spam_words"]
sel = tsw[:10][::-1]  # 取最区分度的前10，倒序便于条形从下往上
names = [r["feature"].replace("word_freq_", "").replace("char_freq_%", "字符 ").replace("%3B", ";").replace("%28", "(").replace("%5B", "[").replace("%21", "!").replace("%24", "$").replace("%23", "#") for r in sel]
gaps = [r["gap"] for r in sel]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.barh(names, gaps, color=OI["vermillion"])
ax.set_xlabel("诈骗邮件平均出现频率 − 正常邮件平均出现频率")
ax.set_title("诈骗短信长这样：大写堆、free、remove、000")
despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_word_gap.png")); plt.close(fig)

# ---------- fig6：大写连续长度特征对诈骗的区分度 ----------
cap_keys = ["capital_run_length_total", "capital_run_length_longest", "capital_run_length_average"]
word_keys = ["word_freq_free", "word_freq_remove", "word_freq_000"]
pool = {r["feature"]: r for r in tsw}
rows = cap_keys + word_keys
disp = ["大写长度总和", "最长连续大写", "平均连续大写", "free", "remove", "000"]
spam_m = [pool[k]["spam_mean"] for k in rows]
ham_m = [pool[k]["ham_mean"] for k in rows]
y_pos = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.barh(y_pos - 0.2, spam_m, height=0.4, color=OI["vermillion"], label="诈骗邮件")
ax.barh(y_pos + 0.2, ham_m, height=0.4, color=OI["skyblue"], label="正常邮件")
ax.set_yticks(y_pos); ax.set_yticklabels(disp)
ax.set_xlabel("平均出现频率")
ax.set_title("大写全靠吼：三个大写长度特征是最强信号")
despine(ax); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_capital_features.png")); plt.close(fig)

# ---------- fig7：线性核与 RBF 核在 2D 的边界形状对比 ----------
k2 = S["pca"]["kernel_2d"]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, (kname, title) in zip(axes, [("linear", "线性核"), ("rbf", "RBF 核")]):
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                colors=[OI["skyblue"], OI["orange"]], alpha=0.16)
    ax.contour(xx, yy, Z, levels=[0], colors=[OI["black"]], linewidths=1.8)
    ax.scatter(pca_X[y == 0, 0], pca_X[y == 0, 1], s=8, c=OI["skyblue"], alpha=0.5, edgecolors="none")
    ax.scatter(pca_X[y == 1, 0], pca_X[y == 1, 1], s=8, c=OI["vermillion"], alpha=0.5, edgecolors="none")
    ax.set_title(f"{title}（2D CV {k2[kname]['cv_mean_pp']:.1f}%）")
    ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2")
    despine(ax)
fig.suptitle("核技巧：RBF 把点映射到高维，边界不再是直线", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_pca_kernels.png")); plt.close(fig)

print("M12 figures 已生成:", os.listdir(FIG))
