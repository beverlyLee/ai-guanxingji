# -*- coding: utf-8 -*-
"""
M11 配图脚本：从 ../stats.json 与 ../data 重生 7 张科学可视化配图。
风格：Okabe-Ito 色板、去 top/right 脊线、dpi 150/300、PingFang SC 中文。
运行：python code/figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
TMP = os.path.join(ROOT, "_tmp")
os.makedirs(TMP, exist_ok=True)
os.environ["TMPDIR"] = TMP

plt.rcParams.update({
    "font.family": "PingFang SC",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Okabe-Ito 色盲安全色板
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

with open(os.path.join(ROOT, "stats.json"), encoding="utf-8") as f:
    S = json.load(f)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- 图1：单棵决策树过拟合 ----------
def fig_overfit():
    ds = S["depth_sweep"]
    depths = [("None" if d == "None" else d) for d in ds["depth"]]
    x = list(range(len(depths)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, ds["train_acc"], "o-", color=OI["vermillion"], lw=2.2, label="训练集准确率")
    ax.plot(x, ds["cv_acc"], "s-", color=OI["bluish"], lw=2.2, label="交叉验证准确率")
    ax.fill_between(x, ds["cv_acc"], ds["train_acc"], color=OI["vermillion"], alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("决策树最大深度 max_depth")
    ax.set_ylabel("准确率")
    ax.set_ylim(0.6, 1.02)
    ax.axvline(len(depths) - 2, color=OI["grey"], ls="--", lw=1)
    ax.annotate("不限制深度：训练集 100%\n验证集却掉到 76.5%",
                xy=(len(depths) - 1, 1.0), xytext=(2.5, 0.9),
                fontsize=10.5, color=OI["vermillion"],
                arrowprops=dict(arrowstyle="->", color=OI["vermillion"]))
    ax.legend(frameon=False, loc="lower left")
    despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_overfit.png"))
    plt.close(fig)


# ---------- 图2：随机森林 / Bagging 原理示意 ----------
def fig_rf_concept():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # 原始样本
    ax.add_patch(FancyBboxPatch((0.2, 6.2), 1.6, 2.4, boxstyle="round,pad=0.06",
                                fc="#F2F2F2", ec=OI["black"]))
    ax.text(1.0, 7.4, "原始样本\n(303人)", ha="center", va="center", fontsize=10.5)
    # Bootstrap 抽样箭头
    for i, bx in enumerate([3.0, 4.3, 5.6, 6.9]):
        ax.add_patch(FancyArrowPatch((1.9, 7.4), (bx - 0.1, 7.4),
                                     arrowstyle="->", color=OI["grey"], lw=1.3))
    # 4个自助样本
    for i, bx in enumerate([3.0, 4.3, 5.6, 6.9]):
        ax.add_patch(FancyBboxPatch((bx, 6.0), 1.0, 1.0, boxstyle="round,pad=0.05",
                                    fc=OI["skyblue"], ec=OI["black"]))
        ax.text(bx + 0.5, 6.5, f"样本{i+1}\n(有放回抽样)", ha="center", va="center", fontsize=8.5)
    # 4棵树
    for i, bx in enumerate([3.0, 4.3, 5.6, 6.9]):
        ax.add_patch(FancyBboxPatch((bx, 3.8), 1.0, 1.2, boxstyle="round,pad=0.05",
                                    fc=OI["bluishgreen"], ec=OI["black"]))
        ax.text(bx + 0.5, 4.4, f"树{i+1}\n(只看部分特征)", ha="center", va="center", fontsize=8.5)
        ax.add_patch(FancyArrowPatch((bx + 0.5, 6.0), (bx + 0.5, 5.0),
                                     arrowstyle="->", color=OI["grey"], lw=1.1))
    # 投票
    ax.add_patch(FancyBboxPatch((3.0, 1.0), 4.9, 1.6, boxstyle="round,pad=0.06",
                                fc=OI["orange"], ec=OI["black"]))
    ax.text(5.45, 1.8, "多数投票 / 平均", ha="center", va="center", fontsize=11, weight="bold")
    for i, bx in enumerate([3.0, 4.3, 5.6, 6.9]):
        ax.add_patch(FancyArrowPatch((bx + 0.5, 3.8), (bx + 0.5, 2.7),
                                     arrowstyle="->", color=OI["grey"], lw=1.1))
    ax.text(5.45, 9.4, "随机森林 = 有放回抽样 + 随机选特征 + 多树投票", ha="center",
            fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_rf_concept.png"))
    plt.close(fig)


# ---------- 图3：OOB 误差随树数量收敛 ----------
def fig_oob_curve():
    curve = S["rf_oob_curve"]
    xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, ys, "o-", color=OI["bluish"], lw=2.2)
    ax.set_xlabel("树的数量 n_estimators")
    ax.set_ylabel("袋外误差 (1 - OOB 准确率)")
    ax.set_ylim(0.12, 0.23)
    ax.axhline(0.1683, color=OI["vermillion"], ls="--", lw=1.2)
    ax.text(210, 0.155, "约 300 棵树后趋于平稳", fontsize=10, color=OI["vermillion"])
    despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_oob_curve.png"))
    plt.close(fig)


# ---------- 图4：特征重要性 ----------
def fig_importance():
    fi = S["random_forest"]["feature_importances"]
    items = sorted(fi.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = [OI["bluishgreen"] if v >= 0.10 else OI["skyblue"] for v in vals]
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("随机森林特征重要性（平均不纯度下降）")
    ax.set_xlim(0, 0.15)
    despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_importance.png"))
    plt.close(fig)


# ---------- 图5：提升算法示意 ----------
def fig_boosting():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    stages = [("第1棵弱树\n先粗略分", 0.8, OI["skyblue"]),
              ("第2棵\n专补上轮错分", 3.4, OI["bluishgreen"]),
              ("第3棵\n再补剩下的", 6.0, OI["orange"])]
    for txt, bx, c in stages:
        ax.add_patch(FancyBboxPatch((bx, 6.2), 2.0, 1.5, boxstyle="round,pad=0.06",
                                    fc=c, ec=OI["black"]))
        ax.text(bx + 1.0, 6.95, txt, ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((2.8, 6.95), (3.35, 6.95), arrowstyle="->",
                                 color=OI["grey"], lw=1.6))
    ax.add_patch(FancyArrowPatch((5.4, 6.95), (5.95, 6.95), arrowstyle="->",
                                 color=OI["grey"], lw=1.6))
    # 权重变化
    ax.add_patch(FancyBboxPatch((0.8, 3.4), 7.2, 1.4, boxstyle="round,pad=0.06",
                                fc="#F2F2F2", ec=OI["black"]))
    ax.text(4.4, 4.1, "被前一轮分错的样本 → 下一轮权重调高 → 被迫重点学习",
            ha="center", va="center", fontsize=10.5)
    # 累加
    ax.add_patch(FancyBboxPatch((0.8, 1.0), 7.2, 1.6, boxstyle="round,pad=0.06",
                                fc=OI["reddish"], ec=OI["black"]))
    ax.text(4.4, 1.8, "最终预测 = 各棵树预测 × 各自权重 累加", ha="center",
            va="center", fontsize=11, weight="bold")
    ax.text(4.4, 9.4, "提升(Boosting)：不是投票，是知错就改、串行纠错", ha="center",
            fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_boosting.png"))
    plt.close(fig)


# ---------- 图6：Stacking 示意 ----------
def fig_stacking():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # 基模型
    for i, (txt, bx) in enumerate([("随机森林", 1.0), ("梯度提升", 4.0)]):
        ax.add_patch(FancyBboxPatch((bx, 6.2), 2.2, 1.4, boxstyle="round,pad=0.06",
                                    fc=OI["skyblue"], ec=OI["black"]))
        ax.text(bx + 1.1, 6.9, txt, ha="center", va="center", fontsize=10.5)
        ax.text(bx + 1.1, 6.5, "输出预测概率", ha="center", va="center", fontsize=8.5)
    # 元模型
    ax.add_patch(FancyBboxPatch((2.6, 3.4), 2.6, 1.4, boxstyle="round,pad=0.06",
                                fc=OI["orange"], ec=OI["black"]))
    ax.text(3.9, 4.1, "元学习器\n(逻辑回归)", ha="center", va="center", fontsize=10.5)
    for bx in [2.1, 5.1]:
        ax.add_patch(FancyArrowPatch((bx, 6.2), (3.9, 4.8), arrowstyle="->",
                                     color=OI["grey"], lw=1.4))
    ax.add_patch(FancyBboxPatch((2.6, 1.0), 2.6, 1.4, boxstyle="round,pad=0.06",
                                fc=OI["bluishgreen"], ec=OI["black"]))
    ax.text(3.9, 1.7, "最终预测", ha="center", va="center", fontsize=11, weight="bold")
    ax.add_patch(FancyArrowPatch((3.9, 3.4), (3.9, 2.4), arrowstyle="->",
                                 color=OI["grey"], lw=1.4))
    ax.text(4.4, 9.4, "Stacking：让模型再当一次学生，学怎么把基模型的输出拼起来",
            ha="center", fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_stacking.png"))
    plt.close(fig)


# ---------- 图7：模型对比 ----------
def fig_model_compare():
    mc = S["model_comparison"]
    names = [r["model"] for r in mc]
    means = [r["mean"] for r in mc]
    stds = [r["std"] for r in mc]
    order = np.argsort(means)
    names = [names[i] for i in order]; means = [means[i] for i in order]; stds = [stds[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ypos = np.arange(len(names))
    colors = [OI["bluish"] if m >= 0.81 else OI["skyblue"] for m in means]
    ax.barh(ypos, means, xerr=[s * 1.96 for s in stds], color=colors,
            capsize=4, error_kw=dict(ecolor=OI["grey"]))
    for i, m in enumerate(means):
        ax.text(m + 0.004, i, f"{m:.3f}", va="center", fontsize=9)
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_xlabel("5 折交叉验证准确率（误差棒 = 95% 置信区间）")
    ax.set_xlim(0.7, 0.88)
    ax.axvline(0.765, color=OI["vermillion"], ls=":", lw=1.2)
    ax.text(0.766, len(names) - 0.3, "单棵深树基线", fontsize=9, color=OI["vermillion"])
    despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_model_compare.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_overfit()
    fig_rf_concept()
    fig_oob_curve()
    fig_importance()
    fig_boosting()
    fig_stacking()
    fig_model_compare()
    print("7 张配图已生成 ->", FIG)
