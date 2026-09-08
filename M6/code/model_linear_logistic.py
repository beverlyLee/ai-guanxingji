#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M6 实验脚本：手写梯度下降，把线性回归和逻辑回归从公式跑到真实数据上。

数据集：UCI Wine Quality（红葡萄酒，1599 条，11 个理化指标 + 质量分 quality 0-10）。
  - 线性回归：用理化指标「猜」连续的质量分 quality（单特征 / 多特征 / 非线性多项式）。
  - 逻辑回归：把 quality>=7 定义为「好酒」(good, 二分类)，用同样 11 个指标预测 P(good)。

所有数字都会写进 ../stats.json，供正文与配图引用（三处一致）。
图由本脚本从计算出的数组直接生成（一图一事）。

运行（受管 python env，已含 numpy/matplotlib/pandas/scikit-learn）：
  python code/model_linear_logistic.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260908)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data", "winequality-red.csv")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "STHeiti", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
C = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]


def mse(y, yp):
    return float(np.mean((y - yp) ** 2))


def r2_score(y, yp):
    return float(1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2))


# ---------- 读数据 ----------
df = pd.read_csv(DATA, sep=";")
FEATS = list(df.columns[:-1])  # 11 个理化指标
X_raw = df[FEATS].values.astype(float)
y_quality = df["quality"].values.astype(float)
# 逻辑回归二分类标签：quality>=7 算好酒
y_good = (df["quality"] >= 7).astype(int).values

# ================= 一、线性回归（手写梯度下降） =================
Xz, Xmu, Xsd = standardize(X_raw)  # 标准化，便于梯度下降稳定
Xb = add_bias(Xz)

# ---- 1a 单特征：quality ~ alcohol ----
idx_alc = FEATS.index("alcohol")
xa = Xz[:, [idx_alc]]
xab = add_bias(xa)
w1 = RNG.normal(0, 0.01, 2)
lr_slow, lr_good, lr_big = 0.005, 0.1, 0.5
hist_lr = {}
for lr in (lr_slow, lr_good, lr_big):
    w = RNG.normal(0, 0.01, 2)
    h = []
    for _ in range(4000):
        yp = xab @ w
        grad = (2.0 / len(xab)) * (xab.T @ (yp - y_quality))
        w = w - lr * grad
        h.append(mse(y_quality, xab @ w))
    hist_lr[str(lr)] = h
w1 = RNG.normal(0, 0.01, 2)
for _ in range(4000):
    yp = xab @ w1
    w1 = w1 - lr_good * (2.0 / len(xab)) * (xab.T @ (yp - y_quality))
single_mse = mse(y_quality, xab @ w1)
single_r2 = r2_score(y_quality, xab @ w1)
# 转回原始尺度：quality = (w1[1]/sd)*alcohol + (w1[0] - w1[1]*mu/sd)
w_orig_alc = w1[1] / Xsd[idx_alc]
b_orig_alc = w1[0] - w1[1] * Xmu[idx_alc] / Xsd[idx_alc]

# ---- 1b 多特征：quality ~ 全部 11 个指标 ----
w_multi = RNG.normal(0, 0.01, Xb.shape[1])
for _ in range(6000):
    yp = Xb @ w_multi
    w_multi = w_multi - lr_good * (2.0 / len(Xb)) * (Xb.T @ (yp - y_quality))
multi_mse = mse(y_quality, Xb @ w_multi)
multi_r2 = r2_score(y_quality, Xb @ w_multi)
# 原始尺度系数
w_orig_multi = w_multi[1:] / Xsd
b_orig_multi = w_multi[0] - np.sum(w_multi[1:] * Xmu / Xsd)

# ---- 1c 非线性：alcohol 的多项式，看训练/测试 R2 随阶数变化（过拟合）----
alc_raw = X_raw[:, idx_alc]
n = len(alc_raw)
perm = RNG.permutation(n)
cut = int(0.7 * n)
tr, te = perm[:cut], perm[cut:]
poly_deg = [1, 2, 3, 4, 5, 6]
poly_train_r2, poly_test_r2 = [], []
for d in poly_deg:
    Phi_tr = np.column_stack([alc_raw[tr] ** p for p in range(1, d + 1)])
    Phi_te = np.column_stack([alc_raw[te] ** p for p in range(1, d + 1)])
    Phitr_z, m, s = standardize(Phi_tr)
    Phite_z = (Phi_te - m) / s
    A = add_bias(Phitr_z)
    w = np.linalg.lstsq(A, y_quality[tr], rcond=None)[0]
    poly_train_r2.append(r2_score(y_quality[tr], A @ w))
    B = add_bias(Phite_z)
    poly_test_r2.append(r2_score(y_quality[te], B @ w))

# ---- 优化设置演示：未标准化时同样 lr 直接发散 ----
Xb_raw = add_bias(X_raw)
w_raw = RNG.normal(0, 0.01, Xb_raw.shape[1])
loss_unscaled = None
diverged = False
for _ in range(2000):
    yp = Xb_raw @ w_raw
    w_raw = w_raw - 0.05 * (2.0 / len(Xb_raw)) * (Xb_raw.T @ (yp - y_quality))
    if not np.all(np.isfinite(Xb_raw @ w_raw)):
        diverged = True
        break
loss_unscaled = mse(y_quality, Xb_raw @ w_raw) if not diverged else float("nan")

print("=" * 60)
print("线性回归（手写梯度下降，特征已标准化，lr=0.1）")
print(f"  单特征 alcohol:  R2={single_r2:.3f}  MSE={single_mse:.3f}")
print(f"    原始尺度方程: quality = {w_orig_alc:.3f} * alcohol + {b_orig_alc:.3f}")
print(f"  多特征(11):     R2={multi_r2:.3f}  MSE={multi_mse:.3f}")
print(f"  多项式各阶 训练R2={[round(r,3) for r in poly_train_r2]}")
print(f"  多项式各阶 测试R2={[round(r,3) for r in poly_test_r2]}")
print(f"  未标准化+lr=0.05 发散? {diverged} (末损失={loss_unscaled})")

# ================= 二、逻辑回归（手写梯度下降） =================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 训练/测试切分（与多项式同一份切分，保证可比）
Xb_tr, Xb_te = Xb[tr], Xb[te]
y_tr, y_te = y_good[tr], y_good[te]

def bce(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

w_log = RNG.normal(0, 0.01, Xb.shape[1])
lr_log = 0.1
log_loss_hist = []
for _ in range(6000):
    z = Xb_tr @ w_log
    p = sigmoid(z)
    grad = (1.0 / len(Xb_tr)) * (Xb_tr.T @ (p - y_tr))
    w_log = w_log - lr_log * grad
    if _ % 200 == 0:
        log_loss_hist.append(bce(y_tr, sigmoid(Xb_tr @ w_log)))
p_tr = sigmoid(Xb_tr @ w_log)
p_te = sigmoid(Xb_te @ w_log)
train_acc = float(np.mean((p_tr >= 0.5).astype(int) == y_tr))
val_acc = float(np.mean((p_te >= 0.5).astype(int) == y_te))

from sklearn.metrics import roc_auc_score
auc = float(roc_auc_score(y_te, p_te))

# 最高概率样本（"模型说这瓶大概率是好酒"）
top_i = int(np.argmax(p_te))
top_prob = float(p_te[top_i])
top_feats = {FEATS[j]: float(X_raw[te][top_i, j]) for j in range(len(FEATS))}

# 对照 sklearn，验证手写正确
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=5000).fit(Xb_tr, y_tr)
auc_sk = float(roc_auc_score(y_te, clf.predict_proba(Xb_te)[:, 1]))

# sigmoid 曲线（沿 alcohol 维度投影，用于 fig4）
alc_grid = np.linspace(alc_raw.min(), alc_raw.max(), 200)
phi = (alc_grid - Xmu[idx_alc]) / Xsd[idx_alc]
z_grid = w_log[0] + w_log[1 + idx_alc] * phi
sigm_grid = sigmoid(z_grid)

print("-" * 60)
print("逻辑回归（手写梯度下降，lr=0.1，good=quality>=7）")
print(f"  正类占比(好酒) = {y_good.mean():.3f}")
print(f"  训练准确率={train_acc:.3f}  测试准确率={val_acc:.3f}  AUC={auc:.3f}")
print(f"  sklearn 对照 AUC={auc_sk:.3f}")
print(f"  最高概率样本 P(good)={top_prob:.3f}, alcohol={top_feats['alcohol']}, volatile={top_feats['volatile acidity']}")
print("=" * 60)

# ================= 写 stats.json =================
stats = {
    "meta": {
        "dataset": "UCI Wine Quality (red)",
        "n": int(n),
        "features": FEATS,
        "target_linear": "quality (3-8)",
        "target_logistic": "good = (quality>=7)",
        "positive_rate": float(y_good.mean()),
        "split": "70/30, seed=20260908",
    },
    "linear_single": {
        "feature": "alcohol",
        "r2": single_r2, "mse": single_mse,
        "w_std": float(w1[1]), "b_std": float(w1[0]),
        "w_orig": float(w_orig_alc), "b_orig": float(b_orig_alc),
        "x_min": float(alc_raw.min()), "x_max": float(alc_raw.max()),
        "line_x": [float(alc_raw.min()), float(alc_raw.max())],
        "line_y": [float(w_orig_alc * alc_raw.min() + b_orig_alc),
                   float(w_orig_alc * alc_raw.max() + b_orig_alc)],
        "scatter_alcohol": [float(v) for v in alc_raw],
        "scatter_quality": [float(v) for v in y_quality],
    },
    "linear_multi": {
        "r2": multi_r2, "mse": multi_mse,
        "coef_std": [float(v) for v in w_multi[1:]],
        "coef_orig": [float(v) for v in w_orig_multi],
        "intercept_orig": float(b_orig_multi),
    },
    "linear_poly": {
        "degrees": poly_deg,
        "train_r2": [float(v) for v in poly_train_r2],
        "test_r2": [float(v) for v in poly_test_r2],
    },
    "linear_lr": {
        "lrs": [lr_slow, lr_good, lr_big],
        "mse_hist": {str(k): [float(v) for v in hist_lr[str(k)]] for k in (lr_slow, lr_good, lr_big)},
        "unscaled_diverged": bool(diverged),
        "unscaled_final_loss": (None if diverged else float(loss_unscaled)),
    },
    "logistic": {
        "positive_rate": float(y_good.mean()),
        "train_acc": train_acc, "val_acc": val_acc, "auc": auc,
        "auc_sklearn": auc_sk,
        "final_train_loss": float(bce(y_tr, sigmoid(Xb_tr @ w_log))),
        "loss_hist": [float(v) for v in log_loss_hist],
        "weights_std": [float(v) for v in w_log[1:]],
        "intercept": float(w_log[0]),
        "sigmoid_alcohol": [float(v) for v in alc_grid],
        "sigmoid_prob": [float(v) for v in sigm_grid],
        "top_sample": {"prob": top_prob, "features": top_feats},
        "scatter_alcohol": [float(v) for v in alc_raw[te]],
        "scatter_good": [int(v) for v in y_te],
        "scatter_prob": [float(v) for v in p_te],
    },
}
with open(os.path.join(ROOT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("stats.json 已写入")

# ================= 配图（一图一事） =================
# fig1：单特征线性回归拟合线
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.scatter(alc_raw, y_quality, s=12, alpha=0.35, color=C[1], label="每瓶酒")
xs = np.array([alc_raw.min(), alc_raw.max()])
ax.plot(xs, w_orig_alc * xs + b_orig_alc, color=C[5], lw=2.5,
        label=f"拟合直线 (R2={single_r2:.2f})")
ax.set_xlabel("酒精度 alcohol (% vol)")
ax.set_ylabel("质量分 quality")
ax.set_title("线性回归第一步：用一条直线去「猜」质量分")
ax.legend(frameon=False, loc="upper left"); despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_linear_fit.png")); plt.close(fig)

# fig2：学习率对收敛速度的影响（标准化后）
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for lr, col in zip((lr_slow, lr_good, lr_big), (C[2], C[0], C[6])):
    h = hist_lr[str(lr)]
    ax.plot(range(1, len(h) + 1), h, color=col, lw=2, label=f"lr={lr}")
ax.set_xlabel("梯度下降迭代次数")
ax.set_ylabel("均方误差 MSE")
ax.set_title("学习率是梯度下降的油门：太小慢慢爬，太大晃半天")
ax.legend(frameon=False); despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_lr_convergence.png")); plt.close(fig)

# fig3：多项式阶数 vs 训练/测试 R2（过拟合）
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(poly_deg, poly_train_r2, marker="o", color=C[1], lw=2, label="训练集 R2")
ax.plot(poly_deg, poly_test_r2, marker="s", color=C[5], lw=2, label="测试集 R2")
best_d = int(np.argmax(poly_test_r2))
ax.axvline(best_d, ls="--", color=C[3], alpha=0.8)
ax.text(best_d + 0.05, min(poly_test_r2) + 0.02, f"测试最佳阶数={best_d}", color="#555")
ax.set_xlabel("多项式阶数（只含 alcohol 的幂）")
ax.set_ylabel("R2")
ax.set_title("非线性回归：阶数越高训练越准，但测试会先涨后跌")
ax.legend(frameon=False); despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_poly_overfit.png")); plt.close(fig)

# fig4：逻辑回归 sigmoid 概率曲线
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for g in (0, 1):
    m = np.array(y_te) == g
    ax.scatter(alc_raw[te][m], p_te[m], s=14, alpha=0.4,
               color=(C[2] if g == 1 else C[6]),
               label=("好酒(good=1)" if g == 1 else "非好酒(good=0)"))
ax.plot(alc_grid, sigm_grid, color=C[0], lw=2.5, label="模型给出的 P(good)")
ax.axhline(0.5, ls="--", color="#888", lw=1)
ax.set_xlabel("酒精度 alcohol (% vol)")
ax.set_ylabel("P(好酒)")
ax.set_title("逻辑回归把线性打分变成 0~1 的概率")
ax.legend(frameon=False, loc="upper left"); despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_logistic_sigmoid.png")); plt.close(fig)

# fig5：逻辑回归权重（哪些指标把酒推向「好酒」）
order = np.argsort(-np.abs(w_log[1:]))
names = [FEATS[i] for i in order]
vals = [w_log[1:][i] for i in order]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
cols = [C[0] if v > 0 else C[5] for v in vals]
ax.barh(range(len(names)), vals, color=cols)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.axvline(0, color="#333", lw=1)
ax.set_xlabel("逻辑回归权重（对数几率尺度）")
ax.set_title("哪些理化指标把酒推向「好酒」？酒精度最关键")
despine(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_logistic_weights.png")); plt.close(fig)

print("5 张图已生成：fig1_linear_fit / fig2_lr_convergence / fig3_poly_overfit / fig4_logistic_sigmoid / fig5_logistic_weights")
