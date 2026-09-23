#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CV4 配图：从 experiment.py 落盘的 .npz / stats.json 重生 7 张图。
风格：Heiti SC 中文、Okabe-Ito 色盲安全配色、dpi300、去脊线、无 chart junk。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDATA = os.path.join(ROOT, "figures_data")
FIGOUT = os.path.join(ROOT, "figures")
os.makedirs(FIGOUT, exist_ok=True)

with open(os.path.join(ROOT, "stats.json"), encoding="utf-8") as f:
    STATS = json.load(f)

# 字体
zh = "Heiti SC" if any("Heiti SC" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams["font.family"] = [zh, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Okabe-Ito
C = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def no_spines(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def save(fig, name):
    p = os.path.join(FIGOUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------------------------------
# fig1: MNIST 样本 + 类别分布
# ---------------------------------------------------------------------------
def fig1():
    d = np.load(os.path.join(FIGDATA, "mnist_samples.npz"))
    samples = d["samples"]  # 10 x 5 x 28 x 28
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    # 左：样本网格
    ax = axes[0]
    grid = np.zeros((10 * 28, 5 * 28), dtype=np.uint8)
    for r in range(10):
        for c in range(5):
            grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = samples[r, c]
    ax.imshow(grid, cmap="gray")
    ax.set_title("MNIST：每个数字挑 5 个真实样本", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    # 右：类别分布
    ax = axes[1]
    cc = STATS["mnist"]["class_counts"]
    xs = [str(k) for k in sorted(cc)]
    ys = [cc[k] for k in sorted(cc)]
    ax.bar(xs, ys, color=C[2])
    ax.set_title("训练集 10 类数量（共 %d 张）" % STATS["mnist"]["n_train"], fontsize=12)
    ax.set_xlabel("数字类别"); ax.set_ylabel("样本数")
    no_spines(ax)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig1_mnist.png")


# ---------------------------------------------------------------------------
# fig2: Otsu 阈值直方图
# ---------------------------------------------------------------------------
def fig2():
    d = np.load(os.path.join(FIGDATA, "card.npz"))
    hist = d["hist"]; t = int(d["otsu_t"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(256), hist / hist.max(), width=1.0, color=C[5], alpha=0.85)
    ax.axvline(t, color=C[6], lw=2.5, label="Otsu 阈值 = %d" % t)
    ax.set_title("卡面数字面板灰度直方图的双峰与自动阈值", fontsize=12)
    ax.set_xlabel("灰度值"); ax.set_ylabel("归一化频数")
    ax.legend(frameon=False)
    no_spines(ax)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig2_otsu.png")


# ---------------------------------------------------------------------------
# fig3: 信用卡定位结果
# ---------------------------------------------------------------------------
def fig3():
    d = np.load(os.path.join(FIGDATA, "card.npz"))
    card = d["card"]; det = d["det_boxes"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(card, cmap="gray")
    for (x, y, w, h) in det:
        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor=C[6], facecolor="none", lw=1.8))
    ax.set_title("CV1 流水线定位到的数字框（阈值+形态学+轮廓）", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, "fig3_card_detect.png")


# ---------------------------------------------------------------------------
# fig4: 形态学开运算对比（停车位前景提取）
# ---------------------------------------------------------------------------
def fig4():
    d = np.load(os.path.join(FIGDATA, "parking_morph.npz"))
    bin_img = d["bin"]; opened = d["opened"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(bin_img, cmap="gray")
    axes[0].set_title("Otsu 阈值后（含孤立噪点）", fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(opened, cmap="gray")
    axes[1].set_title("形态学开运算后（去噪、保留整块）", fontsize=12)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    save(fig, "fig4_morphology.png")


# ---------------------------------------------------------------------------
# fig5: 停车位识别 + 精度召回
# ---------------------------------------------------------------------------
def fig5():
    d = np.load(os.path.join(FIGDATA, "parking.npz"))
    img = d["sample_img"]; overlay = d["sample_overlay"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("生成的停车场示意图", fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(overlay, cmap="gray")
    axes[1].set_title("定位到的车辆（白框）", fontsize=12)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    save(fig, "fig5_parking_detect.png")

    # 精度召回条
    p = STATS["parking"]
    fig2, ax = plt.subplots(figsize=(6, 3.4))
    names = ["精确率", "召回率", "F1"]
    vals = [p["precision"], p["recall"], p["f1"]]
    bars = ax.bar(names, vals, color=[C[2], C[3], C[6]])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.2f" % v, ha="center", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("30 帧聚合：车位占用判别（阈值+形态学+轮廓）", fontsize=12)
    no_spines(ax); ax.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    save(fig2, "fig5b_pr.png")


# ---------------------------------------------------------------------------
# fig6: SIFT 关键点
# ---------------------------------------------------------------------------
def fig6():
    d = np.load(os.path.join(FIGDATA, "panorama.npz"))
    img1 = d["img1"]; img2 = d["img2"]; kp1 = d["kp1_pt"]; kp2 = d["kp2_pt"]
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(img1, cmap="gray" if img1.ndim == 2 else None)
    axes[0].scatter(kp1[:, 0], kp1[:, 1], s=4, c=C[6], alpha=0.7)
    axes[0].set_title("图1 关键点 %d 个" % len(kp1), fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(img2, cmap="gray" if img2.ndim == 2 else None)
    axes[1].scatter(kp2[:, 0], kp2[:, 1], s=4, c=C[1], alpha=0.7)
    axes[1].set_title("图2（由图1变换而来）关键点 %d 个" % len(kp2), fontsize=12)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    save(fig, "fig6_sift.png")


# ---------------------------------------------------------------------------
# fig7: 全景拼接 + 内点匹配
# ---------------------------------------------------------------------------
def fig7():
    d = np.load(os.path.join(FIGDATA, "panorama.npz"))
    stitched = d["stitched"]; img1 = d["img1"]
    match = d["match_pts"]; inl = d["inlier_mask"]
    if stitched.ndim == 3:
        stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(img1, cmap="gray" if img1.ndim == 2 else None)
    axes[0].set_title("图1（基准）", fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(stitched, cmap="gray" if stitched.ndim == 2 else None)
    axes[1].set_title("拼接结果（RANSAC 还原单应后对齐）", fontsize=12)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    save(fig, "fig7_panorama.png")


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("all figures done")


if __name__ == "__main__":
    import cv2  # noqa
    main()
