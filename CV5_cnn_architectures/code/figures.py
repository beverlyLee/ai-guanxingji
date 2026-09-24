# -*- coding: utf-8 -*-
"""
CV5 配图：全部由 Python 从 stats.json / 架构定义重生，不依赖任何外部图片。

风格：STHeiti 中文、Okabe-Ito 色盲安全配色、figure.dpi=150、savefig.dpi=300、
去脊线、无 chart junk（沿用 M15 写法）。

    fig1_macro       参数爆炸与精度跃迁（参数 vs 年份，top1 标注）
    fig2_lenet       LeNet-5 结构示意图（CNN 模板）
    fig3_vgg_stack   3×3 小核堆叠等效 7×7（参数更省）
    fig4_inception   Inception 多分支 + 1×1 瓶颈
    fig5_resnet      残差块结构 + 退化曲线（ResNet 的核心）
    fig6_rf          感受野随层增长：VGG16 vs ResNet-50
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import rf_track, vgg_rf_seq, resnet_rf_seq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

stats = json.loads((ROOT / "stats.json").read_text(encoding="utf-8"))
A = stats["architectures"]
DG = stats["degradation"]
RF = stats["receptive_field"]

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


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig] {p.name}")


def box(ax, x, y, w, h, text, col, fs=8.0, tc="white", lw=0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor=OK["dark"] if lw else "none", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, col=OK["dark"], lw=1.2, ms=11):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=col, lw=lw))


# ---------------------------------------------------------------- fig1 宏观
def fig1():
    keys = ["LeNet-5", "AlexNet", "VGG16", "GoogLeNet(Inception-v1)", "ResNet-50"]
    years = ["LeNet-5", "AlexNet", "VGG16", "GoogLeNet", "ResNet-50"]
    yr = [1998, 2012, 2014, 2014, 2015]
    params = [A[n]["params_M"] for n in keys]
    top1 = [A[n]["imagenet_top1"] for n in keys]
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    bars = ax.bar(x, params, width=0.62, color=[OK["grey"], OK["blue"], OK["vermillion"], OK["green"], OK["purple"]])
    ax.set_yscale("log")
    ax.set_ylabel("参数量（百万，对数轴）")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({y})" for n, y in zip(years, yr)], fontsize=10)
    ax.set_ylim(0.05, 400)
    for xi, p, t in zip(x, params, top1):
        ax.text(xi, p * 1.12, f"{p:.3g}M", ha="center", va="bottom", fontsize=9.5, color=OK["dark"])
        if t is not None:
            ax.text(xi, 0.075, f"top1 {t*100:.1f}%", ha="center", va="bottom", fontsize=9, color=OK["dark"])
    ax.text(0.012, 0.97, "参数量先爆炸（VGG 1.38 亿）后靠结构收敛，ImageNet top1 却一路向上。",
            transform=ax.transAxes, fontsize=11, color=OK["dark"], va="top")
    ax.text(0.012, 0.86, "注：LeNet-5 参数量为 MNIST 模型估算；top1 仅适用于 ImageNet 模型。",
            transform=ax.transAxes, fontsize=8.5, color=OK["grey"], va="top")
    save(fig, "fig1_macro.png")


# ---------------------------------------------------------------- fig2 LeNet-5
def fig2():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.6); ax.axis("off")
    n = 8
    w = (12.5 - 0.24 - (n - 1) * 0.18) / n
    gap = 0.18
    boxes = [
        ("输入\n1×32×32\n灰度图", OK["grey"]),
        ("C1 卷积\n1→6, 5×5\n无填充", OK["green"]),
        ("S2 池化\n2×2 / 2\n6×14×14", OK["sky"]),
        ("C3 卷积\n6→16, 5×5", OK["green"]),
        ("S4 池化\n2×2 / 2\n16×5×5", OK["sky"]),
        ("C5 卷积\n16→120, 5×5\n1×1", OK["green"]),
        ("F6 全连接\n120→84\n+ tanh", OK["vermillion"]),
        ("输出层\n84→10\n+ 高斯连接", OK["orange"]),
    ]
    for i, (text, col) in enumerate(boxes):
        x = 0.12 + i * (w + gap)
        box(ax, x, 1.7, w, 1.9, text, col, fs=8.2)
        if i < n - 1:
            arrow(ax, x + w, 2.65, x + w + gap, 2.65)
    ax.text(0.12, 1.05,
            "LeNet-5（1998）定义 CNN 模板：卷积提特征 → 池化降采样 → 全连接分类。"
            " 全模型仅 6.2 万参数，靠反向传播端到端训练。",
            fontsize=10.5, color=OK["dark"])
    ax.text(0.12, 0.45, "此后 14 年里，所有后来的网络都在反复回答一个问题：卷积之后，怎么堆叠才更好？",
            fontsize=10, color=OK["grey"])
    save(fig, "fig2_lenet.png")


# ---------------------------------------------------------------- fig3 VGG 小核堆叠
def fig3():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.8); ax.axis("off")
    # 左：单个 7×7 卷积
    box(ax, 0.4, 1.7, 2.6, 1.8, "7×7 卷积\nC→C", OK["vermillion"], fs=9.5)
    ax.text(1.7, 1.45, "感受野 = 7×7", ha="center", fontsize=9.5, color=OK["dark"])
    ax.text(1.7, 1.05, r"参数量 = 7²·C² = 49C²", ha="center", fontsize=10, color=OK["dark"])
    # 右：三个 3×3 堆叠
    bx = 4.0
    for i in range(3):
        box(ax, bx + i * 2.1, 1.7, 1.8, 1.8, f"3×3\n卷积", OK["green"], fs=9.5)
        if i < 2:
            arrow(ax, bx + i * 2.1 + 1.8, 2.65, bx + (i + 1) * 2.1, 2.65)
    ax.text(bx + 1.05 * 2.1 - 0.15, 1.05, r"3 层堆叠 → 感受野 = 1+3×2 = 7×7", ha="center", fontsize=9.5, color=OK["dark"])
    ax.text(bx + 1.05 * 2.1 - 0.15, 0.62, r"参数量 = 3×(3²·C²) = 27C²", ha="center", fontsize=10, color=OK["dark"])
    # 箭头从 7×7 指向堆叠，标注更省
    arrow(ax, 3.1, 2.65, bx, 2.65, col=OK["dark"])
    C = 64
    ax.text(5.75, 3.55, "同样看到 7×7，", ha="center", fontsize=10, color=OK["dark"])
    ax.text(5.75, 3.2, f"参数却从 {49*C*C:,} 降到 {27*C*C:,}（省 {100*(1-27/49):.0f}%）",
            ha="center", fontsize=10, color=OK["vermillion"])
    ax.text(0.12, 4.35, "VGG 的洞见：用小卷积堆叠替代大卷积，感受野不变，参数更少，还能叠更多非线性（ReLU）。",
            fontsize=11, color=OK["dark"])
    save(fig, "fig3_vgg_stack.png")


# ---------------------------------------------------------------- fig4 Inception
def fig4():
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_xlim(0, 12.0); ax.set_ylim(0, 5.4); ax.axis("off")
    # 输入
    box(ax, 0.3, 2.2, 1.4, 1.3, "输入\n特征图", OK["grey"], fs=9)
    ix = 0.3 + 1.4
    # 四条分支
    branches = [
        ("1×1\n卷积", OK["blue"], 2.1),
        ("1×1 降维\n→ 3×3 卷积", OK["green"], 3.7),
        ("1×1 降维\n→ 5×5 卷积", OK["green"], 5.3),
        ("3×3 池化\n→ 1×1", OK["sky"], 6.9),
    ]
    bw = 1.5
    for txt, col, x in branches:
        arrow(ax, ix, 2.85, x, 2.85)
        box(ax, x, 2.2, bw, 1.3, txt, col, fs=8.6)
    # concat
    cx = 6.9 + bw + 0.3
    arrow(ax, 6.9 + bw, 2.85, cx, 2.85)
    box(ax, cx, 2.2, 1.9, 1.3, "拼接\n(通道合并)", OK["purple"], fs=9)
    ax.text(cx + 0.95, 1.9, "1×1 瓶颈把通道压下来，\n再喂给 3×3/5×5，参数骤减", ha="center", fontsize=9, color=OK["dark"])
    # 参数对比
    C = 256
    naive = (3*3 + 5*5) * C * C
    with_bn = (1*1 + 1*1*C*(C//4) + 3*3*(C//4)*(C//4) + 1*1*C*(C//4) + 1*1*C*(C//4) + 5*5*(C//4)*(C//4))
    ax.text(0.3, 1.15, f"若直接 3×3+5×5：约 {naive:,} 参数", fontsize=10, color=OK["vermillion"])
    ax.text(0.3, 0.75, f"加 1×1 瓶颈后：约 {with_bn:,} 参数（约降到 1/{(naive/with_bn):.1f}）", fontsize=10, color=OK["dark"])
    ax.text(0.3, 4.9, "Inception：同一层并行多种尺度，让网络自己学“该看多大”；1×1 是其中的算力阀门。",
            fontsize=11, color=OK["dark"])
    save(fig, "fig4_inception.png")


# ---------------------------------------------------------------- fig5 ResNet
def fig5():
    fig = plt.figure(figsize=(12.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.28)

    # (a) 残差块
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 6.0); ax1.set_ylim(0, 5.2); ax1.axis("off")
    box(ax1, 0.2, 2.4, 1.3, 1.2, "x", OK["grey"], fs=11)
    box(ax1, 1.8, 2.4, 1.7, 1.2, "权重层\n(Conv-BN-ReLU\n-Conv-BN)", OK["green"], fs=8.4)
    arrow(ax1, 1.5, 3.0, 1.8, 3.0)
    # shortcut
    ax1.add_patch(Rectangle((1.55, 2.0), 4.0, 2.0, facecolor="none", edgecolor=OK["orange"], lw=1.4, ls="--"))
    box(ax1, 4.0, 2.4, 1.6, 1.2, "F(x)+x\nΣ", OK["orange"], fs=9)
    arrow(ax1, 3.5, 3.0, 4.0, 3.0, col=OK["green"])
    arrow(ax1, 1.5, 2.7, 4.0, 2.7, col=OK["orange"], lw=1.0, ms=9)
    arrow(ax1, 5.6, 3.0, 5.9, 3.0)
    box(ax1, 5.9, 2.4, 0.0, 1.2, "", OK["grey"])  # placeholder no-op
    ax1.text(5.75, 3.0, "→", fontsize=14, color=OK["dark"], ha="center", va="center")
    ax1.text(3.55, 4.6, "y = F(x) + x", fontsize=12, color=OK["dark"], ha="center")
    ax1.text(3.55, 0.5, "短路连接让梯度直达浅层：要学“残差 F(x)”，\n而非直接学映射，深层也能训得动。",
             fontsize=9.5, color=OK["dark"], ha="center")

    # (b) 退化曲线（用最终训练误差做柱状对比）
    ax2 = fig.add_subplot(gs[0, 1])
    groups = ["plain", "ResNet"]
    depths = [20, 56]
    plain = [DG["plain_20_layer_train_err"], DG["plain_56_layer_train_err"]]
    resn = [DG["resnet_20_layer_train_err"], DG["resnet_56_layer_train_err"]]
    x = np.arange(len(depths))
    w = 0.34
    b1 = ax2.bar(x - w/2, [p*100 for p in plain], w, color=OK["vermillion"], label="plain（无残差）")
    b2 = ax2.bar(x + w/2, [r*100 for r in resn], w, color=OK["green"], label="ResNet（有残差）")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d} 层" for d in depths])
    ax2.set_ylabel("CIFAR-10 训练误差 (%)")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for bars in (b1, b2):
        for r in bars:
            ax2.text(r.get_x()+r.get_width()/2, r.get_height()+0.15, f"{r.get_height():.2f}",
                     ha="center", fontsize=8.5, color=OK["dark"])
    ax2.text(0.5, 0.62, "plain：56 层(9.97%) 比 20 层(7.24%) 还差\nResNet：56 层(6.41%) 反超 20 层(7.05%)",
             transform=ax2.transAxes, fontsize=9, color=OK["dark"], ha="center",
             bbox=dict(boxstyle="round", fc="#fff7e6", ec=OK["orange"], lw=0.8))
    ax2.set_ylim(0, 12)
    save(fig, "fig5_resnet.png")


# ---------------------------------------------------------------- fig6 感受野
def fig6():
    vgg = rf_track(vgg_rf_seq())
    res = rf_track(resnet_rf_seq())
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(range(len(vgg)), vgg, "-o", color=OK["vermillion"], lw=2, ms=4, label="VGG16（16 卷积层）")
    ax.plot(range(len(res)), res, "-o", color=OK["green"], lw=2, ms=4, label="ResNet-50（49 卷积层）")
    ax.set_xlabel("卷积/池化层序号（按前向顺序）")
    ax.set_ylabel("累计感受野（像素）")
    ax.legend(frameon=False, fontsize=9.5)
    ax.annotate(f"VGG16 末层感受野≈{vgg[-1]}\n（但只有 16 个卷积层）", xy=(len(vgg)-1, vgg[-1]),
                xytext=(len(vgg)-9, vgg[-1]*0.55), fontsize=9, color=OK["vermillion"],
                arrowprops=dict(arrowstyle="-", color=OK["vermillion"], lw=1))
    ax.annotate(f"ResNet-50 末层感受野≈{res[-1]}\n（49 个卷积层，靠步长跳得更快）", xy=(len(res)-1, res[-1]),
                xytext=(len(res)-16, res[-1]*0.7), fontsize=9, color=OK["green"],
                arrowprops=dict(arrowstyle="-", color=OK["green"], lw=1))
    ax.set_title("同样的 224×224 输入，层数越多、步长越大，末层神经元“一眼看到”的范围越大",
                 fontsize=11, loc="left", pad=8)
    save(fig, "fig6_rf.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done")
