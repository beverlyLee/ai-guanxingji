#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 配图生成：从 ../stats.json 重生 5 张图（一图一事）。
配色 Okabe-Ito 色盲安全；去脊线；中文用 PingFang SC / STHeiti。
运行（用默认受管 python env，已含 matplotlib+numpy）：
  python code/plot_figures.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

plt.rcParams["font.sans-serif"] = ["PingFang SC", "STHeiti", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# Okabe-Ito 色盲安全
C = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), "..", "stats.json"), encoding="utf-8") as f:
    S = json.load(f)

CLASS_NAMES = S["meta"]["class_names"]


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- fig1：Fashion-MNIST 10 类样本（展示数据集与预处理后样子）----------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
picked = []
for c in range(10):
    idx = np.where(y_test == c)[0][0]
    picked.append(idx)
    ax = axes[c // 5, c % 5]
    ax.imshow(x_test[idx], cmap="gray", vmin=0, vmax=255)
    ax.set_title(CLASS_NAMES[c], fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Fashion-MNIST：10 类服装，每张 28x28 灰度、像素值 0-255", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "fig1_dataset.png"))
plt.close(fig)

# ---------- fig2：学习率对验证准确率的影响 ----------
runs = S["EXP_B_lr"]["runs"]
lrs = S["EXP_B_lr"]["lrs"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, lr in enumerate(lrs):
    acc = runs[str(lr)]["val_acc_hist"]
    ax.plot(range(1, len(acc) + 1), acc, marker="o", ms=3, color=C[i], label=f"lr={lr}")
ax.set_xlabel("训练轮数 (epoch)"); ax.set_ylabel("验证集准确率")
ax.set_title("学习率太大直接崩、太小慢慢爬：验证准确率随轮数变化")
ax.legend(frameon=False); despine(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_learning_rate.png"))
plt.close(fig)

# ---------- fig3：权重初始化方法对比 ----------
ir = S["EXP_C_init"]["runs"]
methods = S["EXP_C_init"]["methods"]
accs = [ir[m]["test_acc"] for m in methods]
labels = ["glorot\n(默认)", "he_normal", "random_normal\n(std=0.05)"]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, accs, color=[C[1], C[2], C[3]])
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=11)
ax.set_ylim(min(accs) - 0.05, max(accs) + 0.03)
ax.set_ylabel("测试集准确率")
ax.set_title("初始化方法对比：三者差异 < 1%，Adam 下选谁都还行")
despine(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_init_method.png"))
plt.close(fig)

# ---------- fig4：初始化标准差的影响 ----------
sr = S["EXP_D_init_std"]["runs"]
stds = S["EXP_D_init_std"]["stds"]
saccs = [sr[str(s)]["test_acc"] for s in stds]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([str(s) for s in stds], saccs, marker="o", color=C[0], lw=2)
for s, a in zip(stds, saccs):
    ax.text(str(s), a + 0.004, f"{a:.3f}", ha="center", fontsize=11)
ax.set_xlabel("初始化标准差 std（random_normal）")
ax.set_ylabel("测试集准确率")
ax.set_title("初始化标准差：0.01 还能稳，再大就崩（0.05/0.1/0.5 全炸）")
despine(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_init_std.png"))
plt.close(fig)

# ---------- fig5：过拟合治理（训练-验证鸿沟被收窄）----------
orun = S["EXP_E_overfit"]["runs"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
groups = [("none", "无正则", C[5]), ("dropout", "Dropout 0.4", C[1]), ("l2", "L2 1e-4", C[2])]
epochs = S["EXP_E_overfit"]["epochs"]
for key, lab, col in groups:
    tr = orun[key]["train_acc_hist"]; va = orun[key]["val_acc_hist"]
    gap = [t - v for t, v in zip(tr, va)]
    ax.plot(range(1, len(gap) + 1), gap, color=col, label=f"{lab}（鸿沟）")
# 标注最终鸿沟，避免标题与数字不一致；值相近时纵向错开防重叠
final_gaps = {key: round(orun[key]["train_acc_hist"][-1] - orun[key]["val_acc_hist"][-1], 3)
              for key, _, _ in groups}
order = ["none", "l2", "dropout"]  # 按值排序：none/l2 接近 → 错开
for i, key in enumerate(order):
    for key2, lab, col in groups:
        if key2 == key:
            y_off = 0.012 * (i - 1)  # -1/0/+1 倍间距
            ax.text(epochs + 1.0, final_gaps[key] + y_off,
                    f"{lab}: {final_gaps[key]:+.3f}", color=col, fontsize=10, va="center")
            break
ax.set_xlim(0, epochs + 8)
ax.set_xlabel("训练轮数 (epoch)"); ax.set_ylabel("训练准确率 - 验证准确率")
ax.set_title("过拟合治理对比：Dropout 把鸿沟压平，L2 1e-4 几乎没动")
ax.legend(frameon=False, loc="upper left"); despine(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_overfit.png"))
plt.close(fig)

print("5 张图已生成：fig1_dataset / fig2_learning_rate / fig3_init_method / fig4_init_std / fig5_overfit")
print("fig3 accs:", dict(zip(methods, accs)))
print("fig4 accs:", dict(zip([str(s) for s in stds], saccs)))
