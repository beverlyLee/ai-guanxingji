#!/usr/bin/env python3
"""
M15 配图：全部由 Python 从 data/ 里的真实数据重生，不依赖任何外部图片。

风格：STHeiti 中文、Okabe-Ito 色盲安全配色、去脊线、无多余装饰。
输出：figures/*.png（300 dpi）

    fig1_arch       网络结构全貌
    fig2_data       数据来源与类别不平衡
    fig3_conv       多通道卷积怎么算
    fig4_ledger     参数账本
    fig5_bn         BatchNorm 前向与推理
    fig6_lrscan     学习率扫描：BN 的价值
    fig7_curves     四组对照曲线
    fig8_confusion  混淆矩阵
    fig9_topconf    误判的结构（依赖 code/analyze_errors.py 的输出）
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
if not FIG.exists():
    FIG.mkdir(parents=True)

# ---------------------------------------------------------------- 样式
plt.rcParams.update({
    "font.sans-serif": ["STHeiti", "PingFang SC", "Heiti TC",
                        "Arial Unicode MS", "SimHei", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "font.size": 10,
})

OK = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "grey": "#999999", "dark": "#333333",
}

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[OK["blue"], OK["vermillion"], OK["green"], OK["orange"], OK["purple"]])

stats = json.loads((DATA / "stats.json").read_text(encoding="utf-8"))
plot = np.load(DATA / "plot_data.npz")
NAMES = stats["class_names"]
RUNS = stats["runs"]
D = stats["dataset"]
LED = stats["ledger"]
NET = stats["net"]
SCAN = stats.get("lr_scan")


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig] {p.name}")


def run_style(r):
    """按运行配置生成颜色/线型/图例文字。"""
    palette = [OK["green"], OK["vermillion"], OK["orange"], OK["purple"], OK["sky"]]
    return palette, r


def sharpest(X, y, cls):
    """从某一类里挑一张最清楚的样本：梯度能量最高的一张。"""
    idx = np.where(y == cls)[0]
    if len(idx) == 0:
        return None
    g = X[idx].astype(np.float32).mean(axis=3)  # 灰度
    e = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)) \
        + np.abs(np.diff(g, axis=2)).mean(axis=(1, 2))
    return int(idx[np.argmax(e)])


# ---------------------------------------------------------------- fig1 网络结构
def fig1():
    fig, ax = plt.subplots(figsize=(13.5, 4.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 4.4); ax.axis("off")

    st = NET["stages"]
    fc = NET["fc"]
    boxes = [("输入\n3×32×32\n3,072 个数", OK["grey"])]
    for s in st:
        boxes.append((f"Conv {s['cin']}→{s['cout']}\n3×3, pad=1\n"
                      f"{s['conv_params']:,} 参数", OK["green"]))
        tail = f"BN {s['bn_params']} 参数\n" if s["bn_params"] else ""
        boxes.append((f"{tail}+ ReLU + 池化2\n→ {s['cout']} 通道 "
                      f"{s['out_hw'][0]}×{s['out_hw'][1]}",
                      OK["sky"]))
    boxes.append((f"展平 → {fc['flat_dim']:,}\nFC {fc['flat_dim']}→{fc['fc_dim']}\n"
                  f"{fc['f1_params']:,} 参数", OK["vermillion"]))
    boxes.append((f"ReLU\nFC {fc['fc_dim']}→{D['n_class']}\n{fc['f2_params']:,} 参数",
                  OK["vermillion"]))
    boxes.append((f"Softmax\n{D['n_class']} 类概率", OK["orange"]))

    n = len(boxes)
    w = (13.5 - 0.24 - (n - 1) * 0.20) / n
    gap = 0.20
    for i, (text, col) in enumerate(boxes):
        x = 0.12 + i * (w + gap)
        ax.add_patch(Rectangle((x, 1.45), w, 1.85, facecolor=col, edgecolor="none"))
        ax.text(x + w / 2, 2.37, text, ha="center", va="center",
                color="white", fontsize=8.0, linespacing=1.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w, 2.37), (x + w + gap, 2.37),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color=OK["dark"], lw=1.1))

    pb = NET["param_breakdown"]
    ax.text(0.12, 0.78,
            f"合计 {pb['total']:,} 个参数：卷积 {pb['conv']:,} + BN {pb['bn']:,}"
            f" + 全连接 {pb['fc']:,}。输入 3,072 个数，输出 {D['n_class']} 个概率。",
            fontsize=10.5, color=OK["dark"])
    ax.text(0.12, 0.30,
            f"训练配置：SGD 动量 {NET['momentum']}，批大小 {NET['batch']}，"
            f"{NET['epochs']} 轮，随机种子 {NET['seed']}；"
            f"训练 {D['n_train']:,} 张，测试 {D['n_test']:,} 张。",
            fontsize=10, color=OK["grey"])
    save(fig, "fig1_arch.png")


# ---------------------------------------------------------------- fig2 数据来源
def fig2():
    X = plot["X_test"]
    y = plot["y_test"]
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.42, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    picked = [0, 1, 2, 3, 7, 9, 13, 14, 17, 18, 25, 26, 28, 33, 38, 40]
    strip = np.zeros((32, 16 * 32 + 15 * 2, 3), dtype=np.uint8)
    x0 = 0
    for c in picked:
        idx = sharpest(X, y, c)
        if idx is None:
            continue
        strip[:, x0:x0 + 32] = X[idx]
        x0 += 34
    ax.imshow(strip, interpolation="nearest")
    ax.set_xticks([i * 34 + 16 for i in range(len(picked))])
    ax.set_xticklabels([NAMES[c] for c in picked], fontsize=8.5, rotation=45,
                       ha="right")
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"(a) GTSRB 真实车载摄像头样本（测试集，32×32 RGB）"
                 f"　共 {D['n_class']} 类、原始训练图 {D['natural_total']:,} 张",
                 fontsize=11, loc="left", pad=8)

    ax2 = fig.add_subplot(gs[1, :])
    cnt = np.array(D["natural_counts"])
    order = np.argsort(cnt)[::-1]
    ax2.bar(range(len(cnt)), cnt[order],
            color=[OK["blue"] if cnt[i] >= cnt.mean() else OK["orange"] for i in order],
            width=0.78)
    ax2.axhline(cnt.mean(), color=OK["dark"], lw=1.0, ls="--")
    ax2.text(len(cnt) - 0.5, cnt.mean() * 1.06, f"均值 {cnt.mean():.0f}",
             ha="right", va="bottom", fontsize=9, color=OK["dark"])
    ax2.annotate(f"最多 {cnt.max():,} 张", xy=(0, cnt.max()),
                 xytext=(2.0, cnt.max() * 1.02), fontsize=9, color=OK["blue"])
    ax2.annotate(f"最少 {cnt.min()} 张", xy=(len(cnt) - 1, cnt.min()),
                 xytext=(len(cnt) - 4.5, cnt.min() + 380), fontsize=9,
                 color=OK["vermillion"],
                 arrowprops=dict(arrowstyle="-", color=OK["vermillion"], lw=1))
    ax2.set_xlabel(f"{D['n_class']} 个类别（按样本数降序）")
    ax2.set_ylabel("该类的原始样本数")
    ax2.set_xticks([])
    ax2.set_title(f"(b) 类别天生不平衡：最多 {cnt.max():,} 张，最少 {cnt.min()} 张，"
                  f"相差 {cnt.max() / cnt.min():.1f} 倍", fontsize=11, loc="left", pad=8)
    save(fig, "fig2_data.png")


# ---------------------------------------------------------------- fig3 多通道卷积
def fig3():
    X = plot["X_test"]
    y = plot["y_test"]
    K = plot["kernels"]  # (cout, 3, 3, cin)
    k = 5
    idx = sharpest(X, y, 14)  # 停车标志：形状干净，方便看清卷积在扫什么
    img = X[idx].astype(np.float32) / 255.0
    kernel = K[k]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.55],
                                          "wspace": 0.26})
    chan_col = {"R": "#C0392B", "G": "#1E8449", "B": "#2471A3"}
    for c, nm in enumerate(["R", "G", "B"]):
        ax = axes[c]
        ax.imshow(img[:, :, c], cmap="gray")
        ax.add_patch(Rectangle((-0.5, -0.5), 9, 9, fill=False,
                               edgecolor=OK["vermillion"], lw=2))
        ax.set_title(f"{nm} 通道，这一小片 3×3\n"
                     f"核值 {kernel[0, 0, c]:+.2f} {kernel[0, 1, c]:+.2f} "
                     f"{kernel[0, 2, c]:+.2f} …", fontsize=8.8,
                     color=chan_col[nm])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ax = axes[3]
    out_map = np.zeros((32, 32))
    # kernel 形状 (k, k, cin)，摊平成 (位置, 通道)，与 im2col 的 (i*k+j)*C+c 顺序一致
    Wf = kernel.reshape(9, 3)
    for i in range(32):
        for j in range(32):
            ii = np.clip(np.arange(i - 1, i + 2), 0, 31)
            jj = np.clip(np.arange(j - 1, j + 2), 0, 31)
            patch = img[np.ix_(ii, jj, [0, 1, 2])].reshape(9, 3)
            out_map[i, j] = (patch * Wf).sum()
    ax.imshow(img, alpha=0.45)
    im = ax.imshow(out_map, cmap="coolwarm", alpha=0.62)
    ax.set_title(f"第 {k} 号核扫完整张图的输出\n（未加偏置，红蓝表示正负响应）",
                 fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label("卷积输出值", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("一个 3×3×3 的卷积核：三个通道各卷一次，对应位置的 27 个数"
                 "逐项相乘再求和，加偏置，得到特征图上的一个数",
                 fontsize=11, y=1.02)
    save(fig, "fig3_conv.png")


# ---------------------------------------------------------------- fig4 参数账本
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.3})

    ax = axes[0]
    vals = [LED["conv1_params"], LED["fc_equiv_params"]]
    labels = [f"一层 3×3 卷积\n({LED['conv1_conv']})",
              f"等价全连接层\n({LED['input_dims']}→{LED['output_dims']:,})"]
    bars = ax.bar(labels, vals, color=[OK["green"], OK["vermillion"]], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("参数个数（对数轴）")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.6, f"{v:,}",
                ha="center", fontsize=10.5, color=OK["dark"])
    ax.set_ylim(1e2, 4e8)
    ax.set_title(f"(a) 同样的输入输出，卷积比全连接省 "
                 f"{LED['fc_equiv_ratio']:,.0f} 倍", fontsize=11, loc="left", pad=8)

    ax = axes[1]
    pb = NET["param_breakdown"]
    segs = [("卷积层", pb["conv"], OK["green"]),
            ("BatchNorm", pb["bn"], OK["sky"]),
            ("全连接层", pb["fc"], OK["vermillion"])]
    left = 0
    for nm, v, col in segs:
        ax.barh([0], [v], left=left, color=col, height=0.45,
                label=f"{nm} {v:,}（{v / pb['total'] * 100:.1f}%）")
        if v / pb["total"] > 0.05:
            ax.text(left + v / 2, 0, f"{v:,}", ha="center", va="center",
                    color="white", fontsize=11)
        left += v
    ax.set_xlim(0, pb["total"] * 1.02)
    ax.set_yticks([])
    ax.set_xlabel("参数个数")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=9)
    ax.set_title(f"(b) 全网络 {pb['total']:,} 个参数里，卷积只占 "
                 f"{pb['conv'] / pb['total'] * 100:.1f}%，大头在最后两层全连接",
                 fontsize=11, loc="left", pad=8)
    save(fig, "fig4_ledger.png")


# ---------------------------------------------------------------- fig5 BN 前向
def fig5():
    fig, ax = plt.subplots(figsize=(13.5, 4.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=10, tc="white"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs)

    def arrow(x1, y1, x2, y2, col=OK["dark"]):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color=col, lw=1.4))

    y0 = 2.9
    box(0.15, y0, 2.0, 0.85, "一层输出\nx (N×C×H×W)", OK["blue"], 9.5)
    arrow(2.15, y0 + 0.42, 2.75, y0 + 0.42)
    box(2.75, y0, 2.3, 0.85, "按通道算这一批的\n均值 μ, 方差 σ²", OK["green"], 9.5)
    arrow(5.05, y0 + 0.42, 5.65, y0 + 0.42)
    box(5.65, y0, 2.4, 0.85, "x̂ = (x − μ) / √(σ² + ε)", OK["green"], 10.5)
    arrow(8.05, y0 + 0.42, 8.65, y0 + 0.42)
    box(8.65, y0, 2.0, 0.85, "y = γ·x̂ + β", OK["orange"], 10.5)
    arrow(10.65, y0 + 0.42, 11.25, y0 + 0.42)
    box(11.25, y0, 2.1, 0.85, "送进 ReLU\n（γ, β 可学习）", OK["grey"], 9.5)

    ax.text(0.15, y0 + 1.16,
            "训练时：μ、σ² 用当前这一批数据现算，同时用滑动平均存一份给推理用",
            fontsize=10, color=OK["dark"])

    y1 = 1.2
    box(0.15, y1, 2.0, 0.85, "推理时\n一次一张图", OK["purple"], 9.5)
    arrow(2.15, y1 + 0.42, 2.75, y1 + 0.42)
    box(2.75, y1, 3.1, 0.85, "用训练期存下的\n滑动平均 μ̂, σ̂²", OK["purple"], 9.5)
    arrow(5.85, y1 + 0.42, 6.45, y1 + 0.42)
    box(6.45, y1, 2.4, 0.85, "同一套 γ, β", OK["purple"], 9.5)
    arrow(8.85, y1 + 0.42, 9.45, y1 + 0.42)
    box(9.45, y1, 2.3, 0.85, "y = γ·x̂ + β", OK["purple"], 10.5)
    arrow(11.75, y1 + 0.42, 12.35, y1 + 0.42)
    box(12.35, y1, 1.0, 0.85, "输出", OK["grey"], 9.5)
    ax.text(0.15, y1 - 0.32,
            "推理时不能再依赖「这一批里恰好有谁」，所以一张图和一批图的输出必须一致",
            fontsize=10, color=OK["purple"])

    fig.suptitle("BatchNorm 前向：训练用当前批统计量，推理用滑动平均",
                 fontsize=12, y=1.0)
    save(fig, "fig5_bn.png")


# ---------------------------------------------------------------- fig6 学习率扫描
def fig6():
    lrs = np.array(SCAN["lrs"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.26})

    ax = axes[0]
    ax.plot(lrs, np.array(SCAN["bn"]) * 100, color=OK["green"], lw=2.2,
            marker="o", ms=6, label="每层卷积后加 BN")
    ax.plot(lrs, np.array(SCAN["nobn"]) * 100, color=OK["vermillion"], lw=2.2,
            marker="s", ms=6, label="不加 BN")
    for x, y in zip(lrs, np.array(SCAN["bn"]) * 100):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=OK["green"])
    for x, y in zip(lrs, np.array(SCAN["nobn"]) * 100):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=OK["vermillion"])
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{v:g}" for v in lrs])
    ax.set_xlabel("学习率（对数轴）")
    ax.set_ylabel("6 轮后的验证准确率（%）")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax.axhline(100 / D["n_class"], color=OK["dark"], lw=0.9, ls=":")
    ax.text(lrs[0], 100 / D["n_class"] + 3, f"瞎猜 {100 / D['n_class']:.1f}%",
            ha="left", fontsize=8.5, color=OK["dark"])
    ax.set_title("(a) 同一份数据、同一个网络，只换学习率和要不要 BN",
                 fontsize=11, loc="left", pad=8)

    ax = axes[1]
    for i, lr in enumerate(lrs):
        hb = SCAN["bn_hist"][i]
        hn = SCAN["nobn_hist"][i]
        ax.plot(range(1, len(hb) + 1), np.array(hb) * 100, color=OK["green"],
                lw=1.7, marker="o", ms=3.4, alpha=0.9 - i * 0.12)
        ax.plot(range(1, len(hn) + 1), np.array(hn) * 100, color=OK["vermillion"],
                lw=1.7, ls="--", marker="s", ms=3.4, alpha=0.9 - i * 0.12)
        ax.annotate(f"lr={lr:g}", (len(hb), hb[-1] * 100),
                    textcoords="offset points", xytext=(4, -3), fontsize=8,
                    color=OK["green"])
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("验证准确率（%）")
    ax.set_ylim(0, 100)
    ax.set_xlim(0.9, 7.6)
    ax.plot([], [], color=OK["green"], lw=1.7, label="加 BN")
    ax.plot([], [], color=OK["vermillion"], lw=1.7, ls="--", label="不加 BN")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("(b) 逐轮看：颜色深浅对应学习率从小到大",
                 fontsize=11, loc="left", pad=8)
    save(fig, "fig6_lrscan.png")


# ---------------------------------------------------------------- fig7 四组对照曲线
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"wspace": 0.24})
    palette = [OK["green"], OK["vermillion"], OK["blue"], OK["orange"]]
    styles = ["-", "--", "-.", ":"]
    marks = ["o", "s", "^", "D"]
    lab = lambda r: ("带 BN" if r["use_bn"] else "不带 BN") + f"，lr={r['lr']:g}"

    ax = axes[0]
    handles = []
    for i, r in enumerate(RUNS):
        ln, = ax.plot(r["hist"]["epoch"][1:], r["hist"]["train_loss"][1:],
                      color=palette[i], ls=styles[i], lw=2, marker=marks[i],
                      ms=3.4, label=lab(r))
        handles.append(ln)
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("训练集损失")
    ax.set_yscale("log")
    ax.set_title("(a) 训练损失（对数轴）：D 组第 11 轮后掉头往上",
                 fontsize=11, loc="left", pad=8)

    ax = axes[1]
    finals = []
    for i, r in enumerate(RUNS):
        ep, acc = r["hist"]["epoch"], r["hist"]["test_acc"]
        ax.plot(ep, acc, color=palette[i], ls=styles[i], lw=2,
                marker=marks[i], ms=3.4)
        finals.append((acc[-1], f"{acc[-1] * 100:.2f}%", palette[i]))
    ax.axhline(1 / D["n_class"], color=OK["dark"], lw=0.9, ls=":")
    ax.text(0.2, 1 / D["n_class"] + 0.035, f"瞎猜 {100 / D['n_class']:.1f}%",
            fontsize=9, color=OK["dark"])
    # 末轮数值标签：按高低排序后上下错开，避免压在一起
    finals.sort()
    gap, prev = 0.055, None
    for y, txt, col in finals:
        yy = y if prev is None else max(y, prev + gap)
        prev = yy
        ax.plot([NET["epochs"], NET["epochs"] + 0.55], [y, yy], color=col, lw=0.8)
        ax.text(NET["epochs"] + 0.7, yy, txt, fontsize=9.5, color=col,
                va="center")
    ax.set_xlim(0, NET["epochs"] + 2.6)
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("测试集准确率")
    ax.set_ylim(0, 1.06)
    ax.set_title("(b) 测试准确率（第 0 轮是没训练时的水平）", fontsize=11,
                 loc="left", pad=8)

    fig.legend(handles, [lab(r) for r in RUNS], loc="lower center", ncol=4,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig7_curves.png")


# ---------------------------------------------------------------- fig8 混淆矩阵
def fig8():
    cm = plot["confusion"].astype(np.float64)
    tot = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmn = cm / tot
    off = cm.copy()
    np.fill_diagonal(off, 0)
    wrong = int(off.sum())

    fig, ax = plt.subplots(figsize=(9.8, 8.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(off == 0, np.ones_like(off)), cmap="Reds",
              vmin=0, vmax=1)
    ax.set_xticks(range(D["n_class"])); ax.set_yticks(range(D["n_class"]))
    ax.set_xticklabels(NAMES, fontsize=6.4, rotation=90)
    ax.set_yticklabels(NAMES, fontsize=6.4)
    ax.set_xlabel("模型预测为")
    ax.set_ylabel("真实类别")
    cb = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.02)
    cb.set_label("该真实类别被判成预测类别的比例（蓝）", fontsize=9)
    r0 = RUNS[0]
    bn = "带 BN" if r0["use_bn"] else "不带 BN"
    ax.set_title(f"{bn}（lr={r0['lr']:g}）在 {D['n_test']:,} 张测试图上的混淆矩阵，"
                 f"总准确率 {r0['acc_final'] * 100:.2f}%\n"
                 f"蓝色对角线是判对的，红格就是判错的 {wrong} 张",
                 fontsize=11.5, pad=12)
    save(fig, "fig8_confusion.png")


# ---------------------------------------------------------------- fig9 误判结构
def fig9():
    ea = json.loads((DATA / "error_analysis.json").read_text(encoding="utf-8"))
    A = ea["A_BN_lr0.01"]
    Dd = ea["D_noBN_lr0.05"]

    fig = plt.figure(figsize=(13.6, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.15, 1.05], wspace=0.42)

    # (a) 错例的构成：同族内 vs 跨族
    ax = fig.add_subplot(gs[0])
    fams = list(A["inner"].keys())
    names_seg = [f"{f}内" for f in fams] + ["跨族"]
    cols = [OK["green"], OK["orange"], OK["sky"], OK["purple"], OK["vermillion"]]
    vals_a = [A["inner"][f] for f in fams] + [A["cross"]]
    vals_d = [Dd["inner"][f] for f in fams] + [Dd["cross"]]

    for row, (vals, tot) in enumerate([(vals_a, A["wrong_total"]),
                                       (vals_d, Dd["wrong_total"])]):
        left = 0.0
        for i, v in enumerate(vals):
            pct = v / tot * 100
            ax.barh([row], [pct], left=left, color=cols[i], height=0.52,
                    label=names_seg[i] if row == 0 else None)
            if v:
                if pct >= 9:
                    ax.text(left + pct / 2, row, f"{v}", ha="center",
                            va="center", color="white", fontsize=9)
                else:
                    ax.text(left + pct / 2, row + 0.34, f"{v}", ha="center",
                            va="bottom", fontsize=8, color=cols[i])
            left += pct

    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"A 带 BN\n共错 {A['wrong_total']} 张",
                        f"D 不带 BN\n共错 {Dd['wrong_total']:,} 张"], fontsize=9)
    ax.set_xlim(0, 195)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("错例占比（%）")
    ax.legend(frameon=False, fontsize=8.5, loc="center right",
              handlelength=1.0, borderaxespad=0.2)
    ax.set_title(f"(a) A 组 {A['wrong_total'] - A['cross']} 张错在同族内部，"
                 f"D 组反过来", fontsize=11, loc="left", pad=10)

    # (b) D 组把图都判成了谁
    ax2 = fig.add_subplot(gs[1])
    top = Dd["pred_of_wrong_top"]
    labs = [d["name"] for d in top][::-1]
    vals = [d["count"] for d in top][::-1]
    ax2.barh(labs, vals, color=OK["vermillion"], height=0.6)
    for i, d in enumerate(top[::-1]):
        ax2.text(d["count"] * 1.18, i, f"{d['count']:,}（{d['share'] * 100:.1f}%）",
                 va="center", fontsize=9, color=OK["dark"])
    ax2.set_xscale("log")
    ax2.set_xlim(1, max(vals) * 12)
    ax2.set_xlabel("被模型判成该类别的错例张数（对数轴）")
    ax2.set_title("(b) D 组 1,975 张错例几乎全塌向「限速30」", fontsize=11,
                  loc="left", pad=10)

    # (c) 限速族的八块牌子，区别只在中间那个数字
    ax3 = fig.add_subplot(gs[2])
    X = plot["X_test"]; y = plot["y_test"]
    speed = [0, 1, 2, 3, 4, 5, 7, 8]
    tw, gap = 32, 4
    tile = np.full((2 * 32 + gap, 4 * tw + 3 * gap, 3), 255, dtype=np.uint8)
    for i, cls in enumerate(speed):
        idx = sharpest(X, y, cls)
        r, c = divmod(i, 4)
        yy, xx = r * (32 + gap), c * (tw + gap)
        tile[yy:yy + 32, xx:xx + 32] = X[idx]
    ax3.imshow(tile, interpolation="nearest")
    ax3.set_xticks([c * (tw + gap) + 16 for c in range(4)])
    ax3.set_xticklabels([NAMES[speed[4 + c]] for c in range(4)], fontsize=9)
    ax3.xaxis.set_ticks_position("bottom")
    ax3.xaxis.set_label_position("bottom")
    for c in range(4):
        ax3.text(c * (tw + gap) + 16, -3.5, NAMES[speed[c]],
                 ha="center", va="bottom", fontsize=9)
    ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_visible(False)
    ax3.set_ylim(2 * 32 + gap + 12, -6)
    ax3.set_anchor("N")
    ax3.set_title("(c) 限速族 8 块牌子，只有中间的数字不同", fontsize=11,
                  loc="left", pad=10)
    save(fig, "fig9_topconf.png")


def main():
    todo = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5,
            "6": fig6, "7": fig7, "8": fig8, "9": fig9}
    only = sys.argv[1:] or list(todo)
    for k in only:
        if k in todo:
            todo[k]()


if __name__ == "__main__":
    main()
