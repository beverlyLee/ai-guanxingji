# -*- coding: utf-8 -*-
"""
CV5 配图：全部由 Python 从 stats.json / 架构定义重生，不依赖任何外部图片。

风格：STHeiti 中文、Okabe-Ito 色盲安全配色、figure.dpi=150、savefig.dpi=300、
去脊线、无 chart junk（沿用 M15 写法）。

    fig1_macro       参数爆炸与精度跃迁（参数 vs 年份，top1 标注）
    fig2_lenet       LeNet-5 算法流程图（全流程 + 每层参数量）
    fig3_vgg_stack   3×3 小核堆叠等效 7×7（参数更省）
    fig4_inception   Inception 多分支 + 1×1 瓶颈
    fig5_resnet      残差块结构 + 退化曲线（ResNet 的核心）
    fig6_rf          感受野随层增长：VGG16 vs ResNet-50
    fig7_alexnet     AlexNet 算法流程图（双卡分组卷积 + ReLU/Dropout/数据增强）
    fig8_vgg         VGG16 算法流程图（5 个 stage + 参数都堆在第一个全连接层）
    fig9_googlenet   GoogLeNet 算法流程图（Inception 模块内部 + 两个辅助分类器）
    fig10_resnet     ResNet-50 算法流程图（4 个 stage × bottleneck + 退化数据对比）

思路：fig7–fig10 都是「横向数据流」图，节点按 [输入, 层, 层, …, 输出] 排一行，
每个节点上方标该层输出张量形状、下方标该段参数量；箭头一律只走水平/垂直两向。
所有数字取自 stats.json（由 code/experiment.py 重算），图内不出现手写数字。
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


def seg(ax, x1, y1, x2, y2, col=OK["dark"], lw=1.2, ls="-"):
    """正交折线的一段（无箭头），只允许水平或垂直。"""
    ax.plot([x1, x2], [y1, y2], color=col, lw=lw, ls=ls, zorder=1,
            solid_capstyle="butt")


def tag(ax, x, y, text, col, tc="white", fs=7.4):
    """小标签（自动圆角框），用于 ReLU / Dropout / 数据增强 等注记。"""
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.35, zorder=5,
            bbox=dict(boxstyle="round,pad=0.30", fc=col, ec="none"))


def layout(widths, gap=0.34, margin=0.28):
    """横向排布：返回每个盒子的左边界 xs 与画布总宽（图宽 = 总宽）。"""
    total = sum(widths) + gap * (len(widths) - 1) + 2 * margin
    xs, x = [], margin
    for w in widths:
        xs.append(x)
        x += w + gap
    return xs, total


def shape_labels(ax, xs, widths, shapes, y, fs=7.5, col=None, va="bottom"):
    for x, w, s in zip(xs, widths, shapes):
        if s:
            ax.text(x + w / 2, y, s, ha="center", va=va, fontsize=fs,
                    color=col or OK["dark"])


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
    """LeNet-5 算法流程图：输入到输出全流程 + 每层参数量。"""
    layers = [
        ("输入\n1×32×32\n灰度图", "1×32×32", "无参数", OK["grey"]),
        ("C1 卷积\n5×5×6, s1\n局部连接\n+ 权重共享", "6×28×28", "参数 156", OK["vermillion"]),
        ("S2 平均池化\n2×2, s2\n平均后加权\n+ 偏置", "6×14×14", "无参数", OK["grey"]),
        ("C3 卷积\n5×5×16, s1", "16×10×10", "参数 2,416", OK["blue"]),
        ("S4 平均池化\n2×2, s2", "16×5×5", "无参数", OK["grey"]),
        ("C5 卷积\n5×5×120, s1\n卷积核刚好\n盖满 5×5", "120×1×1", "参数 48,120", OK["blue"]),
        ("F6 全连接\n120→84\n+ tanh", "84", "参数 10,164", OK["blue"]),
        ("输出全连接\n84→10\nRBF 输出", "10", "参数 850", OK["blue"]),
    ]
    widths = [1.36] * len(layers)
    xs, total = layout(widths, gap=0.40, margin=0.30)
    fig, ax = plt.subplots(figsize=(total, 6.0))
    ax.set_xlim(0, total); ax.set_ylim(0.55, 6.0); ax.axis("off")

    yb, h = 3.22, 1.68
    yc = yb + h / 2
    # 先画箭头，后画盒子：盒子会盖住箭尾，只留箭头在缝里
    for i in range(len(layers) - 1):
        arrow(ax, xs[i] + widths[i], yc, xs[i + 1], yc)
    for i, (txt, shape, prm, col) in enumerate(layers):
        box(ax, xs[i], yb, widths[i], h, txt, col, fs=7.9)
        ax.text(xs[i] + widths[i] / 2, yb + h + 0.12, shape,
                ha="center", va="bottom", fontsize=7.6, color=OK["dark"])
        ax.text(xs[i] + widths[i] / 2, yb - 0.12, prm, ha="center",
                va="top", fontsize=7.6,
                color=OK["grey"] if prm == "无参数" else OK["dark"])

    ax.text(0.30, 5.72, "LeNet-5（1998）算法流程：输入 → 卷积 → 池化 → 卷积 → 池化 → 卷积 → 全连接 → 输出",
            fontsize=11, color=OK["dark"], va="top")
    ax.text(0.30, 5.34, "节点上方 = 该层输出张量形状（C×H×W）；节点下方 = 该层参数量。池化无参数，参数全部长在卷积层和全连接层上。",
            fontsize=8.6, color=OK["grey"], va="top")

    ax.add_patch(Rectangle((0.30, 0.62), total - 0.60, 1.82,
                           facecolor="#fff7e6", edgecolor=OK["orange"], lw=1.0))
    ax.text(0.52, 2.20, "核心创新 · 局部连接 + 权重共享", fontsize=9.8,
            color=OK["vermillion"], va="top")
    ax.text(0.52, 1.86,
            "① 同样要生成 4704 个输出神经元（C1 输出 6@28×28 = 4704），若与 32×32 = 1024 个输入像素全连接，需要 4704×1024 ≈ 480 万个参数；",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 1.50,
            "② 改成 5×5 卷积核：每个神经元只连 5×5 的局部区域，同一个核在整张图上共享权重 —— C1 只有 25×6 个权重 + 6 个偏置 = 156 个参数。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 1.06,
            "全模型 61,706 个参数（0.0617M），约是 AlexNet 的 1/988；正是这两条假设，让 CNN 第一次变得可训练。MACs 仅 0.0004G。",
            fontsize=8.7, color=OK["dark"], va="top")
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


# ---------------------------------------------------------------- fig7 AlexNet
def fig7():
    """AlexNet 算法流程图：双卡分组卷积 + ReLU / Dropout / 数据增强 三件套。"""
    conv_p = ((11 * 11 * 3 * 96 + 96)
              + (5 * 5 * (96 // 2) * 256 + 256)
              + (3 * 3 * 256 * 384 + 384)
              + (3 * 3 * (384 // 2) * 384 + 384)
              + (3 * 3 * (384 // 2) * 256 + 256))
    fc_p = (9216 * 4096 + 4096) + (4096 * 4096 + 4096) + (4096 * 1000 + 1000)
    assert conv_p + fc_p == A["AlexNet"]["params"], (conv_p + fc_p, A["AlexNet"]["params"])

    nodes = [
        ("输入\n3×227×227", "3×227×227", OK["grey"], 1.00),
        ("Conv1\n11×11, 96\ns4", "96×55×55", OK["blue"], 0.98),
        ("MaxPool\n3×3, s2", "96×27×27", OK["grey"], 0.90),
        ("Conv2\n5×5, 256\ns1, p2", "256×27×27", OK["blue"], 0.98),
        ("MaxPool\n3×3, s2", "256×13×13", OK["grey"], 0.90),
        ("Conv3\n3×3, 384\ns1, p1", "384×13×13", OK["blue"], 0.98),
        ("Conv4\n3×3, 384\ns1, p1", "384×13×13", OK["blue"], 0.98),
        ("Conv5\n3×3, 256\ns1, p1", "256×13×13", OK["blue"], 0.98),
        ("MaxPool\n3×3, s2", "256×6×6", OK["grey"], 0.90),
        ("FC4096\n9216→4096", "4096", OK["blue"], 1.00),
        ("FC4096\n4096→4096", "4096", OK["blue"], 1.00),
        ("FC1000\n4096→1000", "1000", OK["blue"], 0.95),
    ]
    texts = [n[0] for n in nodes]
    shapes = [n[1] for n in nodes]
    cols = [n[2] for n in nodes]
    widths = [n[3] for n in nodes]
    xs, total = layout(widths, gap=0.32, margin=0.30)
    fig, ax = plt.subplots(figsize=(total, 8.10))
    ax.set_xlim(0, total); ax.set_ylim(0, 8.10); ax.axis("off")

    yb, h = 4.45, 1.70
    yc = yb + h / 2

    # (a) 双卡分组背景带（先画，作为底）
    bx0, bx1 = xs[1], xs[7] + widths[7]
    ax.add_patch(Rectangle((bx0, 6.55), bx1 - bx0, 0.90,
                           facecolor=OK["vermillion"], alpha=0.09, lw=0, zorder=0))
    ax.add_patch(Rectangle((bx0, 6.55), bx1 - bx0, 0.90, facecolor="none",
                           edgecolor=OK["vermillion"], lw=1.0, ls="--", zorder=2))
    ax.text(bx0 + 0.14, 7.34, "双卡分组（2 × GTX 580，各 3GB 显存）", fontsize=7.9,
            color=OK["vermillion"], va="top", zorder=3)
    ax.text(bx0 + 0.14, 7.07, "Conv1 / Conv2 / Conv4 / Conv5 各拆成两组，分别放在两张卡上；"
                              "Conv3 跨卡全连接（两组互通）；到全连接层才合并。",
            fontsize=7.9, color=OK["dark"], va="top", zorder=3)
    ax.text(bx0 + 0.14, 6.74, "代价：两组之间不能交流特征；收益：模型终于塞得进当年一张 3GB 的显卡。",
            fontsize=7.9, color=OK["dark"], va="top", zorder=3)

    # (b) 三件套括号（先画线，盒后画）
    seg(ax, xs[1] + widths[1] / 2, 3.98, xs[7] + widths[7] / 2, 3.98, col=OK["orange"])
    seg(ax, xs[1] + widths[1] / 2, 3.98, xs[1] + widths[1] / 2, 3.86, col=OK["orange"])
    seg(ax, xs[7] + widths[7] / 2, 3.98, xs[7] + widths[7] / 2, 3.86, col=OK["orange"])
    seg(ax, xs[9] + widths[9] / 2, 3.98, xs[10] + widths[10] / 2, 3.98, col=OK["orange"])
    seg(ax, xs[9] + widths[9] / 2, 3.98, xs[9] + widths[9] / 2, 3.86, col=OK["orange"])
    seg(ax, xs[10] + widths[10] / 2, 3.98, xs[10] + widths[10] / 2, 3.86, col=OK["orange"])

    # (c) 主数据流箭头
    for i in range(len(nodes) - 1):
        arrow(ax, xs[i] + widths[i], yc, xs[i + 1], yc)

    # (d) 盒子 + 形状/标签
    for i, (txt, col, w) in enumerate(zip(texts, cols, widths)):
        box(ax, xs[i], yb, w, h, txt, col, fs=7.9)
    shape_labels(ax, xs, widths, shapes, yb + h + 0.12, fs=7.5)
    tag(ax, xs[0] + widths[0] / 2, 4.02, "数据增强\n随机裁剪 224×224\n+ 水平翻转",
        OK["orange"], tc=OK["dark"], fs=7.2)
    tag(ax, (xs[1] + xs[7] + widths[7]) / 2, 3.60, "ReLU（每个卷积层后）",
        OK["orange"], tc=OK["dark"], fs=7.4)
    tag(ax, (xs[9] + xs[10] + widths[10]) / 2, 3.60, "Dropout p = 0.5\n(两个 FC4096 之后)",
        OK["orange"], tc=OK["dark"], fs=7.2)

    # (e) 文字区
    ax.text(0.30, 7.96, "AlexNet（2012）算法流程：更深更大的卷积网络首次赢下 ImageNet —— "
                        "双卡分组 + ReLU + Dropout + 数据增强",
            fontsize=11, color=OK["dark"], va="top")
    ax.text(0.30, 7.72, "上方虚线带 = 当年被拆到两张 GPU 上的卷积段；橙色标签 = 三件套。"
                        "节点上方 = 输出张量形状。",
            fontsize=8.5, color=OK["grey"], va="top")

    ax.add_patch(Rectangle((0.30, 0.45), total - 0.60, 2.55,
                           facecolor="#f2f8fc", edgecolor=OK["blue"], lw=1.0))
    ax.text(0.52, 2.84, f"总参数 {A['AlexNet']['params_M']}M = 卷积 {conv_p/1e6:.2f}M + 全连接 {fc_p/1e6:.2f}M；"
                        f"MACs {A['AlexNet']['macs_G']}G；ImageNet top-1 {A['AlexNet']['imagenet_top1']*100:.1f}%。",
            fontsize=9.4, color=OK["dark"], va="top")
    ax.text(0.52, 2.44, "三件套（橙色标签）：① ReLU —— 每个卷积层后使用，比 sigmoid/tanh 收敛快得多；"
                        "② Dropout p=0.5 —— 两个 FC4096 之后，抑制过拟合；",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 2.08, "③ 数据增强 —— 输入侧随机裁剪 224×224 + 水平翻转，等效把数据集扩大若干倍。"
                        "这三样都不是新结构，但缺一不可。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 1.68, f"参数集中在哪：卷积只占 {100*conv_p/A['AlexNet']['params']:.1f}%，"
                        f"两个 FC4096 + 一个 FC1000 吃掉 {100*fc_p/A['AlexNet']['params']:.1f}% —— "
                        "AlexNet 本质仍是「大号全连接分类器」。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 1.26, "注：输入取 227×227、Conv2/Conv4/Conv5 为 groups=2 的分组卷积，"
                        "层序列与尺寸按 code/experiment.py 的 build_alexnet()，与 stats.json 逐项对齐。",
            fontsize=8.2, color=OK["grey"], va="top")
    ax.text(0.52, 0.82, "注意 MaxPool 的重叠池化（3×3 窗口、步长 2）与 11×11 大步长首层 —— "
                        "两者一起把 224 级的输入迅速压到 6×6，才让后面的全连接层不至于彻底爆炸。",
            fontsize=8.2, color=OK["grey"], va="top")
    save(fig, "fig7_alexnet.png")


# ---------------------------------------------------------------- fig8 VGG16
def fig8():
    """VGG16 算法流程图：5 个 stage + 参数其实都堆在第一个全连接层。"""
    stages = [("stage1", [64, 64], "224×224"),
              ("stage2", [128, 128], "112×112"),
              ("stage3", [256, 256, 256], "56×56"),
              ("stage4", [512, 512, 512], "28×28"),
              ("stage5", [512, 512, 512], "14×14")]
    cin, sp = 3, []
    for _, couts, _ in stages:
        p = 0
        for cout in couts:
            p += 9 * cin * cout + cout
            cin = cout
        sp.append(p)
    conv_p = sum(sp)
    fc1 = 25088 * 4096 + 4096
    fc_p = fc1 + (4096 * 4096 + 4096) + (4096 * 1000 + 1000)
    assert conv_p + fc_p == A["VGG16"]["params"], (conv_p + fc_p, A["VGG16"]["params"])

    texts = (["输入\n3×224×224"]
             + [f"{name} · {len(c)} 层\n{len(c)} × conv3-{c[0]}\n输出 {sz}"
                for name, c, sz in stages]
             + ["展平\n7×7×512\n= 25088", "FC4096\n25088\n→ 4096",
                "FC4096\n4096\n→ 4096", "FC1000\n4096\n→ 1000"])
    shapes = ["3×224×224", "224×224×64", "112×112×128", "56×56×256",
              "28×28×512", "14×14×512", "25088", "4096", "4096", "1000"]
    cols = ([OK["grey"], OK["blue"], OK["blue"], OK["blue"], OK["blue"], OK["blue"],
             OK["grey"], OK["vermillion"], OK["blue"], OK["blue"]])
    widths = [0.98, 1.70, 1.70, 1.70, 1.70, 1.70, 1.15, 1.05, 1.05, 0.95]
    xs, total = layout(widths, gap=0.30, margin=0.30)
    fig, ax = plt.subplots(figsize=(total, 8.30))
    ax.set_xlim(0, total); ax.set_ylim(0, 8.30); ax.axis("off")

    yb, h = 4.55, 1.80
    yc = yb + h / 2

    # (a) 5 个 stage 的浅色底（先画）
    for i in range(1, 6):
        ax.add_patch(Rectangle((xs[i] - 0.11, yb - 0.11), widths[i] + 0.22, h + 0.22,
                               facecolor=OK["sky"], alpha=0.18, lw=0, zorder=0))
    # (b) 两段括号
    seg(ax, xs[1] + widths[1] / 2, 6.88, xs[5] + widths[5] / 2, 6.88, col=OK["blue"])
    seg(ax, xs[1] + widths[1] / 2, 6.88, xs[1] + widths[1] / 2, 6.76, col=OK["blue"])
    seg(ax, xs[5] + widths[5] / 2, 6.88, xs[5] + widths[5] / 2, 6.76, col=OK["blue"])
    seg(ax, xs[7] + widths[7] / 2, 6.88, xs[9] + widths[9] / 2, 6.88, col=OK["grey"])
    seg(ax, xs[7] + widths[7] / 2, 6.88, xs[7] + widths[7] / 2, 6.76, col=OK["grey"])
    seg(ax, xs[9] + widths[9] / 2, 6.88, xs[9] + widths[9] / 2, 6.76, col=OK["grey"])
    # (c) 主数据流箭头
    for i in range(len(widths) - 1):
        arrow(ax, xs[i] + widths[i], yc, xs[i + 1], yc)
    # (d) 盒子
    for i in range(len(widths)):
        box(ax, xs[i], yb, widths[i], h, texts[i], cols[i],
            fs=7.8 if cols[i] == OK["vermillion"] else 7.1)
    shape_labels(ax, xs, widths, shapes, 6.55, fs=7.4)
    # (e) 每个 stage 下方：池化 + 该段卷积参数量
    for i in range(1, 6):
        ax.text(xs[i] + widths[i] / 2, 4.28, f"↓ 池化 2×2 / s2 → {stages[i-1][2]}",
                ha="center", va="top", fontsize=7.2, color=OK["grey"])
        ax.text(xs[i] + widths[i] / 2, 3.98, f"卷积参数 {sp[i-1]:,}",
                ha="center", va="top", fontsize=7.4, color=OK["dark"])
    ax.text(xs[6] + widths[6] / 2, 4.28, "↓ 展平", ha="center", va="top",
            fontsize=7.2, color=OK["grey"])

    # (f) 文字区
    ax.text(0.30, 8.16, "VGG16（2014）算法流程：5 个 stage 全部用 3×3 小核，每过一次池化把通道翻倍",
            fontsize=11, color=OK["dark"], va="top")
    ax.text(0.30, 7.92, "浅蓝底 = 5 个 stage（共 13 个卷积层 + 5 次池化）；节点下方 = 该段卷积参数量；FC4096 用高亮色。",
            fontsize=8.5, color=OK["grey"], va="top")
    ax.text(xs[1] + widths[1] / 2, 6.98, "卷积特征提取：13 个卷积层，全部 3×3（stride 1, pad 1）",
            ha="center", fontsize=8.6, color=OK["blue"], va="bottom")
    ax.text(xs[8] + widths[8] / 2, 6.98, "分类器：3 个全连接层",
            ha="center", fontsize=8.6, color=OK["grey"], va="bottom")

    ax.add_patch(Rectangle((0.30, 2.95), total - 0.60, 0.92,
                           facecolor="#fdece0", edgecolor=OK["vermillion"], lw=1.0))
    ax.text(0.52, 3.74, "核心矛盾：VGG16 名义上 16 层，吃参数的其实只有最后那一层全连接。",
            fontsize=9.8, color=OK["vermillion"], va="top")
    ax.text(0.52, 3.36, f"卷积部分合计 {conv_p:,}（约 {conv_p/1e4:.0f} 万，仅占 "
                        f"{100*conv_p/A['VGG16']['params']:.1f}%）；而第一个全连接层 25088×4096 + 4096 = "
                        f"{fc1:,}（约 {fc1/1e8:.3f} 亿），单层就占总参数的 {100*fc1/A['VGG16']['params']:.1f}%。",
            fontsize=8.7, color=OK["dark"], va="top")

    ax.add_patch(Rectangle((0.30, 0.25), total - 0.60, 2.55,
                           facecolor="#f2f8fc", edgecolor=OK["blue"], lw=1.0))
    ax.text(0.52, 2.62, f"总参数 {A['VGG16']['params_M']}M、MACs {A['VGG16']['macs_G']}G、"
                        f"ImageNet top-1 {A['VGG16']['imagenet_top1']*100:.1f}%。",
            fontsize=9.4, color=OK["dark"], va="top")
    ax.text(0.52, 2.22, "VGG 的设计主张只有两条：① 卷积核一律 3×3（stride 1、pad 1，尺寸不缩水）；"
                        "② 每过一次池化就把通道翻倍。结构规整到可以写成一个循环。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 1.82, f"代价：参数几乎全砸在分类器上 —— 13 个卷积层共 {conv_p/1e6:.2f}M"
                        f"（{100*conv_p/A['VGG16']['params']:.1f}%），3 个全连接层吃掉 {fc_p/1e6:.2f}M"
                        f"（{100*fc_p/A['VGG16']['params']:.1f}%）。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 1.42, "一句话总结：VGG16 用「深度 + 小核」赢下了精度，同时把「层数」和「有效参数量」"
                        "彻底脱钩 —— 这个问题交给了下一代。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.52, 0.60, "注：层序列、张量尺寸与参数量按 code/experiment.py 的 build_vgg16() 逐层累加，"
                        "与 stats.json 完全一致。",
            fontsize=8.2, color=OK["grey"], va="top")
    save(fig, "fig8_vgg.png")


# ------------------------------------------------------------- fig9 GoogLeNet
def fig9():
    """GoogLeNet 算法流程图：Inception 模块内部结构 + 两个辅助分类器。"""
    items = [
        ("输入\n3×224×224", OK["grey"], 0.95, "3×224×224"),
        ("stem\n7×7 conv + pool\n1×1 + 3×3\n+ pool", OK["blue"], 1.20, ""),
        ("Inception\n3a", OK["blue"], 0.80, ""),
        ("Inception\n3b", OK["blue"], 0.80, ""),
        ("MaxPool\n3×3, s2", OK["grey"], 0.88, ""),
        ("Inception\n4a", OK["blue"], 0.80, ""),
        ("Inception\n4b", OK["blue"], 0.80, ""),
        ("Inception\n4c", OK["blue"], 0.80, ""),
        ("Inception\n4d", OK["blue"], 0.80, ""),
        ("Inception\n4e", OK["blue"], 0.80, ""),
        ("Inception\n5a", OK["blue"], 0.80, ""),
        ("Inception\n5b", OK["blue"], 0.80, "7×7×1024"),
        ("全局平均\n池化", OK["grey"], 1.00, "1024"),
        ("Dropout", OK["grey"], 0.95, ""),
        ("FC1000\n1000 类", OK["blue"], 0.95, "1000"),
    ]
    texts = [i[0] for i in items]
    cols = [i[1] for i in items]
    widths = [i[2] for i in items]
    shapes = [i[3] for i in items]
    xs, total = layout(widths, gap=0.30, margin=0.28)

    fig, ax = plt.subplots(figsize=(total, 9.70))
    ax.set_xlim(0, total); ax.set_ylim(0, 9.70); ax.axis("off")

    # ---------- 内嵌：Inception 模块内部结构 ----------
    IY0, IY1 = 6.30, 8.80
    ax.add_patch(Rectangle((0.35, IY0), 8.95, IY1 - IY0, facecolor="#f7f7f7",
                           edgecolor=OK["grey"], lw=0.9))
    ax.text(0.55, 8.68, "Inception 模块内部：4 条并行分支，输出在通道维拼接",
            fontsize=9.5, color=OK["dark"], va="top")
    rows = [8.26, 7.81, 7.36, 6.91]        # 自上而下：分支 A / B / C / D
    bh, mid = 0.34, (8.26 + 6.91) / 2
    # 分支第一段（A 直接 1×1；B/C 先 1×1 瓶颈；D 是池化）
    ax.add_patch(Rectangle((1.70, rows[0] - bh / 2), 1.40, bh, facecolor=OK["orange"],
                           edgecolor="none"))
    for y in rows[1:3]:
        ax.add_patch(Rectangle((1.70, y - bh / 2), 1.20, bh, facecolor=OK["orange"],
                               edgecolor="none"))
    ax.add_patch(Rectangle((1.70, rows[3] - bh / 2), 1.20, bh, facecolor=OK["grey"],
                           edgecolor="none"))
    # 分支第二段
    for y in rows[1:]:
        ax.add_patch(Rectangle((3.05, y - bh / 2), 1.10, bh, facecolor=OK["blue"],
                               edgecolor="none"))
    # 主干竖线 + 分支箭头（先画线，后画盒）
    seg(ax, 1.40, mid, 1.52, mid, col=OK["dark"], lw=1.2)
    seg(ax, 1.52, rows[3], 1.52, rows[0], col=OK["dark"], lw=1.2)
    seg(ax, 4.32, rows[3], 4.32, rows[0], col=OK["dark"], lw=1.2)
    for y in rows:
        arrow(ax, 1.52, y, 1.70, y, ms=9)
    for y in rows[1:]:
        arrow(ax, 4.15, y, 4.32, y, ms=9)
    arrow(ax, 3.10, rows[0], 4.32, rows[0], ms=9)
    arrow(ax, 4.32, mid, 4.50, mid, ms=9)
    ax.add_patch(Rectangle((0.50, mid - 0.31), 0.90, 0.62, facecolor=OK["grey"],
                           edgecolor="none"))
    ax.text(0.95, mid, "输入\n特征图", ha="center", va="center", fontsize=7.2,
            color="white", linespacing=1.35)
    for y, t in zip(rows, ["1×1 卷积", "1×1 瓶颈", "1×1 瓶颈", "3×3 池化"]):
        ax.text(2.30 if t != "1×1 卷积" else 2.40, y, t, ha="center", va="center",
                fontsize=7.2, color="white")
    for y, t in zip(rows[1:], ["3×3 卷积", "5×5 卷积", "1×1 卷积"]):
        ax.text(3.60, y, t, ha="center", va="center", fontsize=7.2, color="white")
    ax.add_patch(Rectangle((4.50, mid - 0.60), 1.05, 1.20, facecolor=OK["purple"],
                           edgecolor="none"))
    ax.text(5.03, mid, "通道维\n拼接", ha="center", va="center", fontsize=7.4,
            color="white", linespacing=1.35)
    ax.text(5.78, 8.26, "橙色 = 瓶颈（bottleneck）：", fontsize=7.8,
            color=OK["dark"], va="center")
    ax.text(5.78, 7.88, "先用 1×1 把通道压下来，", fontsize=7.8, color=OK["dark"], va="center")
    ax.text(5.78, 7.50, "再喂给 3×3 / 5×5，算力骤降；", fontsize=7.8, color=OK["dark"], va="center")
    ax.text(5.78, 7.12, "池化分支不做降维。", fontsize=7.8, color=OK["dark"], va="center")
    ax.text(5.78, 6.74, "4 条分支的输出在通道维直接拼在一起。", fontsize=7.8,
            color=OK["dark"], va="center")
    # 右侧说明面板：填满画布右侧空白
    ax.add_patch(Rectangle((9.60, IY0), total - 9.88, IY1 - IY0, facecolor="#fdece0",
                           edgecolor=OK["vermillion"], lw=0.9))
    ax.text(9.82, 8.66, "为什么 Inception 值得单独看", fontsize=9.5,
            color=OK["vermillion"], va="top")
    for k, line in enumerate([
            "① 同一层里同时用 1×1 / 3×3 / 5×5 和池化，网络自己学「该看多大」；",
            "② 1×1 瓶颈先把通道压下来，5×5 分支的算力从约 3.21 亿降到约 4657 万；",
            "③ 不必手工决定「这层该用几×几」，把尺度的选择权交给训练；",
            "④ 代价：模块内部结构变复杂，每个分支的通道数成了新的超参数；",
            "⑤ 参数只有 6.9982M，却比 138M 的 VGG16 更准 —— 结构比规模更重要。"]):
        ax.text(9.82, 8.26 - k * 0.40, line, fontsize=8.4, color=OK["dark"], va="top")


    # ---------- 辅助分类器（先画虚线，后画盒子） ----------
    aux_x = [xs[5] + widths[5] / 2, xs[8] + widths[8] / 2]
    for xc in aux_x:
        ax.add_patch(FancyArrowPatch((xc, 4.55), (xc, 3.76), arrowstyle="-|>",
                                     mutation_scale=10, color=OK["vermillion"], lw=1.1, ls="--"))
    # ---------- 主数据流箭头 ----------
    yb, h = 4.55, 1.40
    yc = yb + h / 2
    for i in range(len(widths) - 1):
        arrow(ax, xs[i] + widths[i], yc, xs[i + 1], yc)
    for i in range(len(widths)):
        box(ax, xs[i], yb, widths[i], h, texts[i], cols[i], fs=7.1)
    shape_labels(ax, xs, widths, shapes, 6.07, fs=7.3)
    for xc in aux_x:
        box(ax, xc - 1.15, 3.05, 2.30, 0.70,
            "辅助分类器 1\n仅训练时监督" if xc == aux_x[0] else "辅助分类器 2\n仅训练时监督",
            OK["vermillion"], fs=7.4)
    ax.text((aux_x[0] + aux_x[1]) / 2, 2.78,
            "辅助分类器（共 2 个）：从 Inception 4a / 4d 引出，只在训练时参与监督、缓解梯度消失，推理时整个丢弃。",
            ha="center", fontsize=8.4, color=OK["vermillion"], va="top")

    # ---------- 文字区 ----------
    ax.text(0.28, 9.54, "GoogLeNet（2014, Inception-v1）算法流程：同一层里并行多种卷积尺度，模块本身就是一个「小网络」",
            fontsize=11, color=OK["dark"], va="top")
    ax.text(0.28, 9.34, "主流程把 Inception 模块画成一个节点，展开看就是上方那个 inset。",
            fontsize=8.4, color=OK["grey"], va="top")

    vgg_fc1 = 25088 * 4096 + 4096
    ax.add_patch(Rectangle((0.28, 0.40), total - 0.56, 2.05,
                           facecolor="#f2f8fc", edgecolor=OK["blue"], lw=1.0))
    ax.text(0.50, 2.28, f"总参数 {A['GoogLeNet(Inception-v1)']['params_M']}M（只有 VGG16 的约 1/20）、"
                        f"MACs {A['GoogLeNet(Inception-v1)']['macs_G']}G（VGG16 的约 1/10）、"
                        f"ImageNet top-1 {A['GoogLeNet(Inception-v1)']['imagenet_top1']*100:.1f}%。",
            fontsize=9.4, color=OK["dark"], va="top")
    ax.text(0.50, 1.90, "① 瓶颈的收益：5×5 分支前面插入一个 1×1 把通道压下来，该分支 MACs 从约 "
                        "3.21 亿降到约 4657 万，只剩原来的 14.5%。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.50, 1.54, f"② 全局平均池化替代最后的全连接层：直接把 7×7×1024 压成 1024 维，省下上千万参数"
                        f"（对比 VGG16 第一个全连接层的 {vgg_fc1/1e8:.3f} 亿）。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.50, 1.18, "③ 瓶颈 + 多尺度并行 + 全局平均池化三招一起，让「更深」第一次不必以「更大」为代价。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.50, 0.78, "注：参数量/MACs 沿用论文公开值（Szegedy et al. 2015，含两个辅助分类器）；"
                        "主流程层序按 Inception-v1 原始配置，数字与 stats.json 一致。",
            fontsize=8.2, color=OK["grey"], va="top")
    save(fig, "fig9_googlenet.png")


# ------------------------------------------------------------- fig10 ResNet-50
def fig10():
    """ResNet-50 算法流程图：4 个 stage × bottleneck + 退化数据对比。"""
    items = [
        ("输入\n3×224×224", OK["grey"], 0.95, "3×224×224"),
        ("Conv1\n7×7, 64\ns2, p3", OK["blue"], 1.05, "112×112×64"),
        ("MaxPool\n3×3, s2\np1", OK["grey"], 1.00, "56×56×64"),
        ("stage1\n3 × bottleneck\n64 → 256", OK["sky"], 1.62, "56×56×256"),
        ("stage2\n4 × bottleneck\n128 → 512", OK["sky"], 1.62, "28×28×512"),
        ("stage3\n6 × bottleneck\n256 → 1024", OK["sky"], 1.62, "14×14×1024"),
        ("stage4\n3 × bottleneck\n512 → 2048", OK["sky"], 1.62, "7×7×2048"),
        ("全局平均\n池化", OK["grey"], 1.05, "2048"),
        ("FC1000\n2048\n→ 1000", OK["blue"], 0.95, "1000"),
    ]
    texts = [i[0] for i in items]
    cols = [i[1] for i in items]
    widths = [i[2] for i in items]
    shapes = [i[3] for i in items]
    xs, total = layout(widths, gap=0.34, margin=0.30)

    fig = plt.figure(figsize=(max(total, 14.40), 9.10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.06, 1.0],
                          width_ratios=[1.34, 1.0], hspace=0.20, wspace=0.15)

    # ---------------- 主流程 ----------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, total); ax.set_ylim(0, 5.30); ax.axis("off")
    yb, h = 2.50, 1.80
    yc = yb + h / 2
    for i in range(3, 7):
        ax.add_patch(Rectangle((xs[i] - 0.10, yb - 0.10), widths[i] + 0.20, h + 0.20,
                               facecolor=OK["sky"], alpha=0.22, lw=0, zorder=0))
    for i in range(len(widths) - 1):
        arrow(ax, xs[i] + widths[i], yc, xs[i + 1], yc)
    for i in range(len(widths)):
        box(ax, xs[i], yb, widths[i], h, texts[i], cols[i], fs=7.2)
    shape_labels(ax, xs, widths, shapes, 4.42, fs=7.4)
    ax.text(0.30, 5.18, "ResNet-50（2015）算法流程：卷积层几乎全部装在残差块里，靠 shortcut 把输入一路加回输出",
            fontsize=11, color=OK["dark"], va="top")
    ax.text(0.30, 4.92, "浅蓝底 = 4 个 stage；节点上方 = 输出张量形状。全网络 49 个卷积层，其中 48 个在 stage 内部"
                        "（16 个 bottleneck 块 × 3 层）。",
            fontsize=8.5, color=OK["grey"], va="top")
    ax.text(0.30, 2.30, "堆叠方式从「层层相加」变成「块块相加」—— 深度由 stage 里的块数决定，而每个块内部始终只有 3 个卷积层。",
            fontsize=8.6, color=OK["dark"], va="top")
    ax.text(0.30, 1.92, f"总参数 {A['ResNet-50']['params_M']}M（只有 VGG16 的约 "
                        f"{100*A['ResNet-50']['params']/A['VGG16']['params']:.0f}%）、"
                        f"MACs {A['ResNet-50']['macs_G']}G、ImageNet top-1 "
                        f"{A['ResNet-50']['imagenet_top1']*100:.1f}%。",
            fontsize=9.4, color=OK["dark"], va="top")
    ax.text(0.30, 1.56, f"退化问题（CIFAR-10 训练误差，He et al. 2016, Fig.1）：普通网络 20 层 "
                        f"{DG['plain_20_layer_train_err']*100:.2f}% → 56 层涨到 {DG['plain_56_layer_train_err']*100:.2f}%"
                        f"（越深越差）；",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.30, 1.20, f"残差网络 20 层 {DG['resnet_20_layer_train_err']*100:.2f}% → 56 层降到 "
                        f"{DG['resnet_56_layer_train_err']*100:.2f}%（越深越好）。不是过拟合，而是深层网络「优化不动」"
                        "—— 加上 shortcut，梯度就有了一条直通浅层的路。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.30, 0.80, "一句话：ResNet 让「更准」只花掉 VGG16 的 18% 参数、约四分之一的算力。",
            fontsize=8.7, color=OK["dark"], va="top")
    ax.text(0.30, 0.46, "注：层序列与尺寸按 code/experiment.py 的 build_resnet50()（每个卷积后接 BN、卷积无偏置），"
                        "参数量与 stats.json 完全一致。",
            fontsize=8.2, color=OK["grey"], va="top")

    # ---------------- 左下：bottleneck 内部 ----------------
    axb = fig.add_subplot(gs[1, 0])
    axb.set_xlim(0, 7.30); axb.set_ylim(0, 3.00); axb.axis("off")
    axb.text(0.05, 2.96, "bottleneck 残差块内部：先 1×1 降维、再 3×3、最后 1×1 升维，shortcut 直连相加",
             fontsize=9.2, color=OK["dark"], va="top")
    by, bh, byc = 0.85, 0.66, 1.18
    # shortcut（先画）
    seg(axb, 0.60, by + bh, 0.60, 2.30, col=OK["orange"], lw=1.6)
    seg(axb, 0.60, 2.30, 6.67, 2.30, col=OK["orange"], lw=1.6)
    arrow(axb, 6.67, 2.30, 6.67, by + bh, col=OK["orange"], lw=1.6, ms=11)
    axb.text(3.60, 2.38, "shortcut（恒等映射）", ha="center", fontsize=8.0,
             color=OK["vermillion"], va="bottom")
    # 主路箭头
    for x0, x1 in [(1.10, 1.45), (2.70, 3.05), (4.30, 4.65), (5.90, 6.25)]:
        arrow(axb, x0, byc, x1, byc)
    # 主路盒子
    for x, w, t, c in [(0.10, 1.00, "x\n256×56×56", OK["grey"]),
                       (1.45, 1.25, "1×1 Conv\n256→64\n+ BN, ReLU", OK["blue"]),
                       (3.05, 1.25, "3×3 Conv\n64→64\n+ BN, ReLU", OK["blue"]),
                       (4.65, 1.25, "1×1 Conv\n64→256\n+ BN", OK["blue"]),
                       (6.25, 0.85, "F(x)+x\nReLU", OK["vermillion"])]:
        box(axb, x, by, w, bh, t, c, fs=7.0)
    axb.text(0.05, 0.78, "通道数与空间尺寸都不变时，shortcut 就是恒等映射，什么都不用做。",
             fontsize=8.1, color=OK["dark"], va="top")
    axb.text(0.05, 0.50, "当通道数不一致、或空间尺寸减半时（stage2–4 的首个块），",
             fontsize=8.1, color=OK["dark"], va="top")
    axb.text(0.05, 0.22, "shortcut 要走一个 1×1 卷积 + BN（projection shortcut）对齐形状。",
             fontsize=8.1, color=OK["dark"], va="top")

    # ---------------- 右下：退化曲线 ----------------
    axd = fig.add_subplot(gs[1, 1])
    plain = [DG["plain_20_layer_train_err"] * 100, DG["plain_56_layer_train_err"] * 100]
    resn = [DG["resnet_20_layer_train_err"] * 100, DG["resnet_56_layer_train_err"] * 100]
    x = np.arange(2)
    wd = 0.34
    b1 = axd.bar(x - wd / 2, plain, wd, color=OK["vermillion"], label="plain（无 shortcut）")
    b2 = axd.bar(x + wd / 2, resn, wd, color=OK["green"], label="ResNet（有 shortcut）")
    axd.set_xticks(x); axd.set_xticklabels(["20 层", "56 层"])
    axd.set_ylabel("CIFAR-10 训练误差 (%)", fontsize=9)
    axd.set_ylim(0, 14.0)
    axd.legend(frameon=False, fontsize=8, loc="upper right")
    for bars in (b1, b2):
        for r in bars:
            axd.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.20,
                     f"{r.get_height():.2f}%", ha="center", fontsize=8, color=OK["dark"])
    axd.set_title("退化数据（CIFAR-10 训练误差，取自 stats.json）",
                  fontsize=8.8, loc="left", pad=6)
    axd.text(0.02, 0.99,
             f"plain 越深越差：{plain[0]:.2f}% → {plain[1]:.2f}%\n"
             f"ResNet 越深越好：{resn[0]:.2f}% → {resn[1]:.2f}%",
             transform=axd.transAxes, fontsize=8.0, color=OK["dark"], ha="left", va="top",
             linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.32", fc="#fff7e6", ec=OK["orange"], lw=0.8))
    save(fig, "fig10_resnet.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    fig7(); fig8(); fig9(); fig10()
    print("done")
