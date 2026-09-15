"""
M13 配图：全部从 stats.json / curves.npz / 原始 CSV 重生，不依赖任何外部图片。
风格：Okabe-Ito 色盲安全配色，PingFang SC 中文，去脊线，无 chart junk。
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
stats = json.loads((ROOT / "stats.json").read_text(encoding="utf-8"))
curves = np.load(ROOT / "curves.npz")

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "font.size": 12,
})

OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#8C8C8C")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- fig1 月均序列
def fig1():
    from experiment import load_raw, ffill
    times, raw = load_raw()
    raw = ffill(raw)
    good = ~np.isnan(raw).any(axis=1)
    raw = raw[good]
    times = [t for t, g in zip(times, good) if g]
    pm = raw[:, 0]
    keys = [(t[0], t[1]) for t in times]
    order = sorted(set(keys))
    mmean = np.array([pm[[i for i, k in enumerate(keys) if k == o]].mean() for o in order])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, mmean, color=OI["blue"], lw=1.8)
    ax.fill_between(x, mmean, mmean.min() * 0.6, color=OI["sky"], alpha=0.18)
    ax.axhline(pm.mean(), color=OI["grey"], ls="--", lw=1,
               label=f"四年均值 {pm.mean():.1f} μg/m³")
    ax.set_xticks(x[::6])
    ax.set_xticklabels([f"{order[i][0]}-{order[i][1]:02d}" for i in range(0, len(order), 6)],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("PM2.5 月均（μg/m³）")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    # 标注最强的两个冬季峰
    top = np.argsort(mmean)[-2:]
    for i in top:
        ax.annotate(f"{mmean[i]:.0f}", (x[i], mmean[i]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9, color=OI["vermillion"])
    ax.set_title("北京古城站 PM2.5 月均：四年里冬季反复冲高", fontsize=13)
    save(fig, "fig1_pm25_monthly.png")


# ---------------------------------------------------------------- fig2 预报时距
def fig2():
    hc = stats["horizon_compare"]
    p1, m1 = hc["h1_persistence"]["r2"], hc["h1_model"]["r2"]
    p24 = stats["baseline_persistence"]["test"]["r2"]
    m24 = stats["best"]["test"]["r2"]
    labels = ["提前 1 小时", "提前 24 小时"]
    persist = [p1, p24]
    model = [m1, m24]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    b1 = ax.bar(x - w / 2, persist, w, color=OI["grey"], label="持续性基线")
    b2 = ax.bar(x + w / 2, model, w, color=OI["blue"], label="神经网络")
    ax.axhline(0, color="#333", lw=0.8)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4 if h >= 0 else -14),
                    ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("测试集 R²")
    ax.set_ylim(-0.25, 1.08)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title("预报越远，基线越先垮，模型才有价值", fontsize=13)
    save(fig, "fig2_horizon.png")


# ---------------------------------------------------------------- fig3 学习率
def fig3():
    lrs = ["0.0001", "0.001", "0.01", "0.1", "0.3", "1.0"]
    colors = [OI["sky"], OI["green"], OI["blue"], OI["orange"], OI["purple"], OI["vermillion"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for lr, c in zip(lrs, colors):
        vac = curves[f"lr_{lr}_val"].astype(float)
        finite = np.isfinite(vac)
        if finite.sum() >= 2:
            ax.plot(np.arange(len(vac))[finite], vac[finite], color=c, lw=1.8,
                    label=f"lr={lr}")
        else:
            ax.plot([0, 1], [vac[0], 3.0], color=c, lw=1.8, label=f"lr={lr}（发散）")
    ax.set_ylim(0.85, 2.0)
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("验证集 MSE（越小越好）")
    ax.legend(frameon=False, fontsize=10, ncol=2)
    ax.set_title("学习率：太小磨蹭，太大发散", fontsize=13)
    ax.annotate("lr=0.3 / 1.0 一轮就炸成 NaN", (20, 1.85), fontsize=10,
                color=OI["vermillion"])
    save(fig, "fig3_lr_sweep.png")


# ---------------------------------------------------------------- fig4 初始化
def fig4():
    d = stats["init_depth"]
    names = [("std_0.01", "固定 std=0.01", OI["grey"]),
             ("xavier", "Xavier", OI["blue"]),
             ("he", "He", OI["green"]),
             ("std_0.5", "固定 std=0.5", OI["vermillion"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for key, label, c in names:
        v = np.array(d[key]["z_std_per_layer"], dtype=float)
        v = np.clip(v, 1e-5, None)
        ax.plot(np.arange(1, len(v) + 1), v, "-o", color=c, lw=1.8, ms=5, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("第几层（预激活 z 的标准差）")
    ax.set_ylabel("σ(z)（对数轴）")
    ax.axhline(1.0, color="#333", ls=":", lw=1)
    ax.annotate("理想：每层都停在 1 附近", (1.05, 1.1), fontsize=10, color="#333")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("五层网络里，初始化标准差把信号传成什么样", fontsize=13)
    save(fig, "fig4_init_depth.png")


# ---------------------------------------------------------------- fig5 dropout
def fig5():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for p, c in [("0.0", OI["grey"]), ("0.2", OI["blue"]), ("0.5", OI["orange"])]:
        tr = curves[f"drop_{p}_train"].astype(float)
        va = curves[f"drop_{p}_val"].astype(float)
        ax.plot(tr, color=c, lw=1.5, ls="--", alpha=0.85)
        ax.plot(va, color=c, lw=2.0, label=f"dropout={p}")
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("MSE")
    ax.set_ylim(0.4, 1.3)
    ax.legend(frameon=False, fontsize=10, title="实线=验证，虚线=训练")
    ax.set_title("dropout：把训练与验证之间的缝收窄", fontsize=13)
    save(fig, "fig5_dropout.png")


# ---------------------------------------------------------------- fig6 L2
def fig6():
    lam = ["0.0", "0.0001", "0.001", "0.01"]
    labels = ["0", "1e-4", "1e-3", "1e-2"]
    wn = [stats["l2"][k]["weight_norm"] for k in lam]
    r2 = [stats["l2"][k]["val"]["r2"] for k in lam]
    x = np.arange(len(lam))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    a1.bar(x, wn, 0.55, color=OI["blue"])
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_xlabel("权重衰减系数 λ"); a1.set_ylabel("权重范数 ‖W‖")
    a1.set_title("L2 把权重收紧", fontsize=12)
    for i, v in enumerate(wn):
        a1.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)
    a2.bar(x, r2, 0.55, color=OI["green"])
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_xlabel("权重衰减系数 λ"); a2.set_ylabel("验证集 R²")
    a2.set_title("验证集随之回升", fontsize=12)
    for i, v in enumerate(r2):
        a2.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)
    save(fig, "fig6_l2.png")


# ---------------------------------------------------------------- fig7 预测对比
def fig7():
    true = curves["best_test_true"]
    pred = curves["best_test_pred"]
    pers = curves["best_test_persist"]
    k = 240
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(np.arange(k), true[:k], color="#333", lw=1.6, label="真实 PM2.5")
    ax.plot(np.arange(k), pred[:k], color=OI["blue"], lw=1.8, label="模型预测（提前一天）")
    ax.plot(np.arange(k), pers[:k], color=OI["grey"], lw=1.2, ls="--", label="持续性基线")
    ax.set_xlabel("测试集小时（连续 240 小时）")
    ax.set_ylabel("PM2.5（μg/m³）")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("提前一天：模型抓住了趋势，跟不上突变", fontsize=13)
    save(fig, "fig7_prediction.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("all figures done")
