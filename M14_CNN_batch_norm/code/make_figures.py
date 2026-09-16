"""
M14 配图生成 (手册 §5)：从 plot_data.npz + stats.json 重生 fig1-4。
风格：matplotlib + numpy；中文 PingFang SC / Heiti SC；Okabe-Ito 色盲安全配色；
      figure.dpi=150, savefig.dpi=300；去脊线、无 chart junk；一图一事。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STSong", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# Okabe-Ito 色盲安全配色
O = "#E69F00"   # orange
B = "#56B4E9"   # sky blue
G = "#009E73"   # bluish green
Y = "#F0E442"   # yellow
Bl = "#0072B2"  # blue
V = "#D55E00"   # vermillion
P = "#CC79A7"   # reddish purple
K = "#000000"

BASE = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/ai-guanxingji/M14_CNN_batch_norm"
d = np.load(f"{BASE}/data/plot_data.npz")
stats = json.load(open(f"{BASE}/data/stats.json"))

loss_bn = d["loss_bn"]
loss_nb = d["loss_nb"]
conv1_out = d["conv1_out"]      # (8,64,64)
filters = d["filters"]          # (8,3,3)
sample_img = d["sample_img"]    # (64,64)
cm_bn = d["cm_bn"]


def save(fig, name):
    fig.savefig(f"{BASE}/figures/{name}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------------
# fig1: 卷积层构造 —— 一张脸 → 8 个卷积核 → 8 张特征图
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 7))
gs = GridSpec(2, 9, figure=fig, height_ratios=[1, 3], hspace=0.25, wspace=0.3)

# 顶行：输入脸 + 8 个卷积核
ax_in = fig.add_subplot(gs[0, 0])
ax_in.imshow(sample_img, cmap="gray", vmin=0, vmax=1)
ax_in.set_title("输入人脸\n(64×64)", fontsize=10)
ax_in.set_xticks([]); ax_in.set_yticks([])
for i in range(8):
    ax = fig.add_subplot(gs[0, i + 1])
    k = filters[i]
    kn = (k - k.min()) / (k.max() - k.min() + 1e-12)
    ax.imshow(kn, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"核{i+1}\n3×3", fontsize=9)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=7)

# 底行：8 张特征图
for i in range(8):
    ax = fig.add_subplot(gs[1, i])
    fmap = conv1_out[i]
    fm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-12)
    ax.imshow(fm, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"特征图 {i+1}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("卷积层的构造：用一个共享的 3×3 小核在整张脸上滑动，抽出 8 种局部特征",
             fontsize=12, y=0.98)
save(fig, "fig1_conv_construction.png")


# ---------------------------------------------------------------------------
# fig2: 参数量对比 (第一层卷积 vs 等价的全连接)
# ---------------------------------------------------------------------------
conv_w = stats["conv1_params"]
fc_w = stats["equiv_fc1_params"]
ratio = stats["conv_param_reduction_x"]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(["全连接层\n(64×64→8)", "卷积层\n(3×3→8)"], [fc_w, conv_w], color=[V, B], width=0.55)
ax.set_yscale("log")
ax.set_ylabel("权重数量（对数刻度）")
ax.set_title("同样的输出，卷积层用权值共享把参数砍掉约 {} 倍".format(int(round(ratio))))
for b, v in zip(bars, [fc_w, conv_w]):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:,}", ha="center", va="bottom", fontsize=10)
ax.text(0.5, fc_w * 0.6, f"减少约 {ratio:.0f}×", color=K, ha="center", fontsize=11,
        bbox=dict(boxstyle="round", fc=Y, ec=K, alpha=0.85))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
save(fig, "fig2_param_compare.png")


# ---------------------------------------------------------------------------
# fig3: 带 / 不带 BatchNorm 的 loss 曲线对比
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
epochs = np.arange(1, len(loss_bn) + 1)
ax.plot(epochs, loss_bn, color=G, lw=2.2, label=f"带 BN (lr=0.01)  终loss={loss_bn[-1]:.3f}")
ax.plot(epochs, loss_nb, color=V, lw=2.2, label=f"不带 BN (lr=0.005)  终loss={loss_nb[-1]:.3f}")
ax.set_xlabel("训练轮次 (epoch)")
ax.set_ylabel("训练集交叉熵损失")
ax.set_title("BatchNormalization：允许更大学习率、收敛更快更稳")
ax.legend(frameon=False, fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
save(fig, "fig3_bn_loss_curve.png")


# ---------------------------------------------------------------------------
# fig4: 带 BN 模型的测试混淆矩阵
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm_bn, cmap="Blues", vmin=0)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["预测: 前20人", "预测: 后20人"], fontsize=10)
ax.set_yticklabels(["实际: 前20人", "实际: 后20人"], fontsize=10)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_bn[i, j]), ha="center", va="center",
                fontsize=14, color="white" if cm_bn[i, j] > cm_bn.max() * 0.5 else K)
acc = stats["acc_with_bn"]
ax.set_title(f"测试混淆矩阵 (准确率 {acc*100:.1f}%)", fontsize=11)
save(fig, "fig4_confusion_matrix.png")

print("all figures done")
