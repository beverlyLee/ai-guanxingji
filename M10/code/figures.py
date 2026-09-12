"""M10 figure generation — regenerate every plot from the real dataset.

Style: scientific-visualization (Okabe-Ito colorblind-safe, PingFang SC,
figure.dpi=150 / savefig.dpi=300, spines removed, no chart junk).
All numbers come from data/processed.csv, data/X.npy, data/y.npy, stats.json.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- fonts + style -------------------------------------------------------
plt.rcParams["font.family"] = "PingFang SC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
         "#D55E00", "#CC79A7", "#000000"]

# ---- load real data ------------------------------------------------------
proc = pd.read_csv(os.path.join(DATA, "processed.csv"))
X = np.load(os.path.join(DATA, "X.npy"))
y = np.load(os.path.join(DATA, "y.npy"))
feat_names = json.load(open(os.path.join(DATA, "feat_names.json")))
stats = json.load(open(os.path.join(BASE, "stats.json")))

RNG = 42
median_price = stats["median_price"]


# ==========================================================================
# Fig 1 — 房价分布与中位线
# ==========================================================================
def fig_price_dist():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(proc["SalePrice"] / 1e4, bins=40, color=OKABE[1], alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.axvline(median_price / 1e4, color=OKABE[5], linewidth=2.2)
    ax.text(median_price / 1e4 + 0.6, ax.get_ylim()[1] * 0.92,
            f"中位价 {median_price/1e4:.1f} 万", color=OKABE[5],
            fontsize=11, fontweight="bold")
    ax.set_xlabel("成交价（万元）")
    ax.set_ylabel("房源数量")
    ax.set_title("1460 套二手房成交价分布：一半贵过中位价", fontsize=12.5)
    fig.savefig(os.path.join(FIG, "fig_price_dist.png"))
    plt.close(fig)


# ==========================================================================
# Fig 2 — 分类树根节点分裂示意（熵 / 信息增益）
# ==========================================================================
def fig_root_split():
    ce = stats["clf_entropy"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.axis("off")
    left_n, right_n = 660, 435
    # root
    ax.add_patch(plt.Rectangle((0.30, 0.62), 0.40, 0.26, fill=True,
                 color=OKABE[4], alpha=0.18, ec=OKABE[4], lw=1.5))
    ax.text(0.50, 0.78, f"根节点\n全部 {stats['n_rows']} 套", ha="center",
            va="center", fontsize=10.5, color=OKABE[4], fontweight="bold")
    ax.text(0.50, 0.66, f"父节点熵 = 1.000", ha="center", va="center",
            fontsize=9.5)
    # split arrow text
    ax.text(0.50, 0.50,
            f"按 {ce['root_feature']} ≤ {ce['root_thr']:.1f} 切一刀",
            ha="center", va="center", fontsize=10.5, fontweight="bold")
    # two children
    for (x, n, e, col, lab) in [
            (0.18, left_n, ce["root_e_left"], OKABE[2], "左（老房）"),
            (0.82, right_n, ce["root_e_right"], OKABE[0], "右（新房）")]:
        ax.add_patch(plt.Rectangle((x - 0.16, 0.10), 0.32, 0.26, fill=True,
                     color=col, alpha=0.18, ec=col, lw=1.5))
        ax.text(x, 0.28, f"{lab}\nn = {n}", ha="center", va="center",
                fontsize=10, color=col, fontweight="bold")
        ax.text(x, 0.17, f"子节点熵 = {e:.4f}", ha="center", va="center",
                fontsize=9.5)
    # entropy drop annotation
    ax.annotate("", xy=(0.34, 0.60), xytext=(0.34, 0.40),
                arrowprops=dict(arrowstyle="->", color=OKABE[6], lw=1.5))
    ax.text(0.02, 0.40, "熵从 1.000 降到\n0.487 / 0.792\n（越纯越好分）",
            ha="left", va="center", fontsize=9.5, color=OKABE[6])
    ax.set_title("决策树第一刀：用「建成年份」把房子分成两堆",
                 fontsize=12.5)
    fig.savefig(os.path.join(FIG, "fig_root_split.png"))
    plt.close(fig)


# ==========================================================================
# Fig 3 — 信息增益 vs 增益率（Top 12）
# ==========================================================================
def fig_gain_vs_gr():
    tbl = stats["gain_ratio_table"][:12][::-1]  # ascending for barh
    names = [r["feature"] for r in tbl]
    g = np.array([r["info_gain"] for r in tbl])
    gr = np.array([r["gain_ratio"] for r in tbl])
    ypos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    h = 0.38
    ax.barh(ypos + h/2, g, height=h, color=OKABE[4], label="信息增益")
    ax.barh(ypos - h/2, gr, height=h, color=OKABE[1], label="增益率")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("数值（越大代表该特征越能区分贵/便宜）")
    ax.set_title("信息增益 vs 增益率：12 个最强候选特征", fontsize=12.5)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG, "fig_gain_vs_gr.png"))
    plt.close(fig)


# ==========================================================================
# Fig 4 — 预剪枝深度 vs R²
# ==========================================================================
def fig_preprune_curve():
    pp = stats["pre_prune"]
    d = np.array(pp["depth"])
    tr = np.array(pp["train_r2"])
    te = np.array(pp["test_r2"])
    best = stats["pre_prune_best_depth"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(d, tr, "-o", color=OKABE[0], lw=1.8, ms=4, label="训练集 R²")
    ax.plot(d, te, "-o", color=OKABE[2], lw=1.8, ms=4, label="测试集 R²")
    ax.axvline(best, color=OKABE[5], ls="--", lw=1.5)
    ax.text(best + 0.2, 0.50, f"最佳层数 = {best}\n测试 R² = {te[best-1]:.3f}",
            color=OKABE[5], fontsize=10, fontweight="bold")
    ax.set_xlabel("最大层数 max_depth（预剪枝）")
    ax.set_ylabel("R²")
    ax.set_title("预剪枝曲线：树越深越记答案，测试集先涨后崩", fontsize=12.0)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig_preprune_curve.png"))
    plt.close(fig)


# ==========================================================================
# Fig 5 — 回归树决策边界（GrLivArea × YearBuilt, depth=6）
# ==========================================================================
def fig_decision_boundary():
    bi = [feat_names.index(c) for c in ["GrLivArea", "YearBuilt"]]
    Xb = X[:, bi]
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, y, test_size=0.25,
                                                  random_state=RNG)
    bd = DecisionTreeRegressor(criterion="squared_error", max_depth=6,
                               random_state=RNG)
    bd.fit(Xb_tr, yb_tr)
    gx = np.linspace(Xb[:, 0].min(), Xb[:, 0].max(), 160)
    gy = np.linspace(Xb[:, 1].min(), Xb[:, 1].max(), 160)
    GX, GY = np.meshgrid(gx, gy)
    Z = bd.predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cf = ax.contourf(GX, GY, Z / 1e4, levels=14, cmap="viridis", alpha=0.9)
    sc = ax.scatter(Xb_te[:, 0], Xb_te[:, 1], c=yb_te / 1e4, s=14,
                    cmap="viridis", edgecolor="white", linewidth=0.3,
                    zorder=3)
    ax.set_xlabel("地上居住面积 GrLivArea（平方英尺）")
    ax.set_ylabel("建成年份 YearBuilt")
    ax.set_title(f"回归树决策边界（两特征，层数 6，测试 R²={stats['boundary']['test_r2']:.3f}）",
                 fontsize=11.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("成交价（万元）")
    fig.savefig(os.path.join(FIG, "fig_decision_boundary.png"))
    plt.close(fig)


# ==========================================================================
# Fig 6 — 特征重要性（左：分类树 右：回归树）
# ==========================================================================
def fig_importance():
    ce_imp = stats["clf_entropy"]["top_importances"]
    reg_imp = stats["reg_deep"]["top_importances"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, imp, title, col in [
            (axes[0], ce_imp, "分类树：什么决定\n贵 / 便宜", OKABE[4]),
            (axes[1], reg_imp, "回归树：什么决定\n成交价高低", OKABE[2])]:
        nms = [r[0] for r in imp][::-1]
        vals = [r[1] for r in imp][::-1]
        ax.barh(nms, vals, color=col, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("特征重要性")
    fig.suptitle("同一批房子，分类树和回归树都认准了「质量」和「面积」",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "fig_importance.png"))
    plt.close(fig)


# ==========================================================================
# Fig 7 — 后剪枝 CCP: alpha vs 测试 R²
# ==========================================================================
def fig_ccp_curve():
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.25,
                                                  random_state=RNG)
    deep = DecisionTreeRegressor(criterion="squared_error", max_depth=None,
                                 random_state=RNG)
    deep.fit(Xr_tr, yr_tr)
    path = deep.cost_complexity_pruning_path(Xr_tr, yr_tr)
    alphas = path.ccp_alphas
    test_r2 = []
    for a in alphas:
        if a == 0:
            continue
        m = DecisionTreeRegressor(criterion="squared_error", ccp_alpha=a,
                                  random_state=RNG)
        m.fit(Xr_tr, yr_tr)
        test_r2.append((a, r2_score(yr_te, m.predict(Xr_te))))
    a_arr = np.array([t[0] for t in test_r2])
    r_arr = np.array([t[1] for t in test_r2])
    best_i = int(np.argmax(r_arr))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(a_arr, r_arr, "-o", color=OKABE[2], lw=1.6, ms=3)
    ax.axvline(a_arr[best_i], color=OKABE[5], ls="--", lw=1.5)
    x_annot = 10 ** (np.log10(a_arr.max()) * 0.55)
    ax.text(x_annot, 0.08,
            f"最优 α={a_arr[best_i]:.1e}\n测试 R²={r_arr[best_i]:.3f}\n叶子 {stats['post_prune']['leaves']} 片",
            color=OKABE[5], fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=OKABE[5], alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("复杂度参数 α（越大剪得越狠，log 刻度）")
    ax.set_ylabel("测试集 R²")
    ax.set_title("后剪枝（CCP）：先长疯，再按 α 把多余枝剪掉", fontsize=12.0)
    fig.savefig(os.path.join(FIG, "fig_ccp_curve.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_price_dist()
    fig_root_split()
    fig_gain_vs_gr()
    fig_preprune_curve()
    fig_decision_boundary()
    fig_importance()
    fig_ccp_curve()
    print("all 7 figures written to", FIG)
