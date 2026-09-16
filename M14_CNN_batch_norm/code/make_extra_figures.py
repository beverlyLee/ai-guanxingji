"""
M14 补充配图：从真实 Olivetti 数据与 run.log 生成 fig5-fig8。
风格同 make_figures.py：matplotlib + numpy，PingFang SC 中文，Okabe-Ito 配色，
figure.dpi=150 / savefig.dpi=300，去脊线，一图一事。
"""
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from sklearn.datasets import fetch_olivetti_faces

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STSong", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

O = "#E69F00"; B = "#56B4E9"; G = "#009E73"; Y = "#F0E442"
Bl = "#0072B2"; V = "#D55E00"; P = "#CC79A7"; K = "#000000"

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


faces = fetch_olivetti_faces(data_home=str(ROOT / "data"))
imgs = faces.images                      # (400, 64, 64)
targets = faces.target                   # (400,)

# ---------------------------------------------------------------------------
# fig5: 数据集样本 —— 4 个人，每人 8 张，展示同一个人的照片也有差异
# ---------------------------------------------------------------------------
n_people, n_each = 4, 8
fig, axes = plt.subplots(n_people, n_each, figsize=(11, 6))
for r in range(n_people):
    who = r
    idx = np.where(targets == who)[0][:n_each]
    for c in range(n_each):
        ax = axes[r, c]
        ax.imshow(imgs[idx[c]], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
    axes[r, 0].set_ylabel(f"第 {who + 1} 人", fontsize=11, rotation=0, ha="right", va="center")
fig.suptitle("数据集长这样：40 个人，每人 10 张 64×64 灰度图，同一个人的照片也各有不同",
             fontsize=12.5, y=1.0)
fig.text(0.5, -0.02, "同一行是同一个人（类内差异），不同行是不同人（类间差异）",
         ha="center", fontsize=10, color="#555555")
save(fig, "fig5_sample_faces.png")

# ---------------------------------------------------------------------------
# fig6: 网络结构图
# ---------------------------------------------------------------------------
stages = [
    ("输入", "1×64×64", B),
    ("卷积 3×3, 8\nReLU\n池化 2×2\nBN", "8×32×32", O),
    ("卷积 3×3, 16\nReLU\n池化 2×2\nBN", "16×16×16", G),
    ("展平", "4096", P),
    ("全连接\nSoftmax", "2", V),
]
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
w = 1.55
for i, (name, shape, col) in enumerate(stages):
    x = 0.25 + i * (w + 0.36)
    box = FancyBboxPatch((x, 1.05), w, 1.05, boxstyle="round,pad=0.06,rounding_size=0.12",
                         linewidth=1.6, edgecolor=col, facecolor=col + "22")
    ax.add_patch(box)
    ax.text(x + w / 2, 1.58, name, ha="center", va="center", fontsize=10.5)
    ax.text(x + w / 2, 0.72, shape, ha="center", va="center", fontsize=9.5, color="#444444")
    if i < len(stages) - 1:
        ar = FancyArrowPatch((x + w + 0.02, 1.58), (x + w + 0.34, 1.58),
                             arrowstyle="-|>", mutation_scale=16, color="#666666", lw=1.4)
        ax.add_patch(ar)
ax.text(5.0, 2.62, "网络结构：两层卷积 + 两层池化，每层后接 BN，最后全连接输出两类",
        ha="center", fontsize=12.5)
ax.text(5.0, 0.22, "总参数 9442 个；第一层卷积只用 88 个参数（80 权重 + 8 偏置）",
        ha="center", fontsize=10, color="#555555")
save(fig, "fig6_arch.png")

# ---------------------------------------------------------------------------
# fig7: 一次卷积怎么算 —— 5×5 输入的一个 3×3 窗口 × 卷积核 → 一个输出数
# ---------------------------------------------------------------------------
x = np.array([
    [0.9, 0.9, 0.2, 0.2, 0.2],
    [0.9, 0.9, 0.2, 0.2, 0.2],
    [0.9, 0.9, 0.2, 0.2, 0.2],
    [0.9, 0.9, 0.2, 0.2, 0.2],
    [0.9, 0.9, 0.2, 0.2, 0.2],
])
k = np.array([[0.5, 0.0, -0.5],
              [0.5, 0.0, -0.5],
              [0.5, 0.0, -0.5]])
win = x[0:3, 0:3]
prod = win * k
val = round(float(prod.sum()), 2)

fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2),
                         gridspec_kw={"width_ratios": [5, 0.6, 3, 3]})

def grid(ax, mat, title, hi=None, fmt="{:.1f}"):
    n, m = mat.shape
    for i in range(n):
        for j in range(m):
            face = "#fff3cd" if (hi and hi[0] <= i < hi[2] and hi[1] <= j < hi[3]) else "white"
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=face,
                                   edgecolor="#bbbbbb", lw=0.8))
            ax.text(j + 0.5, n - 1 - i + 0.5, fmt.format(mat[i, j]),
                    ha="center", va="center", fontsize=9)
    ax.set_xlim(0, m); ax.set_ylim(0, n); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=10.5, pad=6)

grid(axes[0], x, "输入 5×5\n黄框 = 当前窗口", hi=(0, 0, 3, 3))
axes[1].axis("off")
axes[1].text(0.5, 0.5, "×", ha="center", va="center", fontsize=20)
grid(axes[2], k, "卷积核 3×3\n（全图共享）", fmt="{:+.1f}")
grid(axes[3], np.array([[val]]), "输出 5×5\n左上角的值", fmt="{:.2f}")
fig.suptitle("一次卷积就是一次加权求和：窗口里的 9 个数与核的 9 个权重逐项相乘再相加",
             fontsize=12, y=1.16)
save(fig, "fig7_conv_compute.png")

# ---------------------------------------------------------------------------
# fig8: 训练日志截图（终端风格，取自 run.log 的真实输出）
# ---------------------------------------------------------------------------
log_lines = [
    "data: train=300 test=100  (前20人 vs 后20人, 各200张)",
    "input shape: (1, 64, 64)",
    "",
    "[参数量对比] 第一层若用全连接(64x64->8)需 32776 个权重(+8偏置)",
    "[参数量对比] 第一层用卷积(3x3, 权值共享)仅需 80 个权重(+8偏置)",
    "[参数量对比] 卷积相对全连接减少 409.7 倍参数",
    "[参数量对比] 整个网络总参数: 9442",
    "",
    "=== 训练 A: 带 BatchNorm (lr=0.01) ===",
    "  epoch  1/40  train_loss=0.3544  train_acc=0.737",
    "  epoch 11/40  train_loss=0.0146  train_acc=1.000",
    "  epoch 40/40  train_loss=0.0033  train_acc=1.000",
    "  test_acc=0.9500  time=148.7s  final_loss=0.0033",
    "",
    "=== 训练 B: 不带 BatchNorm (lr=0.005) ===",
    "  epoch 40/40  train_loss=0.3464  train_acc=0.570",
    "  test_acc=0.6500  time=151.1s  final_loss=0.3464",
]
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            facecolor="#1e1e1e", edgecolor="#3a3a3a", lw=1.5))
for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    ax.add_patch(plt.Circle((0.035 + i * 0.022, 0.955), 0.011, color=c, zorder=3))
ax.text(0.5, 0.955, "python code/train_cnn.py", ha="center", va="center",
        fontsize=9, color="#888888")
y = 0.90
for line in log_lines:
    col = "#7ee787" if ("test_acc" in line or "[参数量对比]" in line) else "#d5d5d5"
    if line.startswith("==="):
        col = "#ffd866"
    ax.text(0.035, y, line, ha="left", va="top", fontsize=9.2,
            color=col)
    y -= 0.052
save(fig, "fig8_train_log.png")

print("extra figures done")
