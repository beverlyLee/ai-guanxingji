"""
M8 实验脚本：鸢尾花多分类逻辑回归（从原理到决策边界）
数据集：Iris（sklearn 内置，150 条样本，4 个特征，3 个类别各 50 条；源自 UCI Machine Learning Repository）
知识点：逻辑回归原理 / 优化目标(交叉熵) / 梯度计算 / 迭代优化 / 多分类 softmax /
        训练与预测模块 / 决策边界绘制 / 非线性决策边界

所有数字写入 ../stats.json，配图写入 ../figures/，保证正文/脚本/配图三处一致。
运行：python experiment.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

RNG = 42
np.random.seed(RNG)

# 中文字体（macOS）
for cand in ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# Okabe-Ito 色盲安全配色
C = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00",
    "purple": "#CC79A7", "black": "#000000", "grey": "#999999",
}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- 数据与切分
iris = load_iris()
X_all = iris.data                      # (150, 4)
y = iris.target                        # 0/1/2
FEATURES = iris.feature_names
TARGETS = list(iris.target_names)      # setosa / versicolor / virginica
CN = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]

# 保存一份 csv 便于复现
np.savetxt(os.path.join(ROOT, "data", "iris.csv"),
           np.column_stack([X_all, y]), delimiter=",",
           header=",".join([f.replace(" ", "_") for f in FEATURES]) + ",target",
           comments="", fmt="%.4f")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.30, stratify=y, random_state=RNG)

scaler = StandardScaler().fit(X_tr)
Xtr = scaler.transform(X_tr)
Xte = scaler.transform(X_te)

N_TR, N_TE = len(y_tr), len(y_te)

# ------------------------------------------------- 手写多分类逻辑回归（softmax）
K = 3


def onehot(yy, k=K):
    Y = np.zeros((len(yy), k))
    Y[np.arange(len(yy)), yy] = 1.0
    return Y


def add_bias(A):
    return np.c_[np.ones(len(A)), A]


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)      # 数值稳定
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(Xb, W, Y):
    P = softmax(Xb @ W)
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))


def analytical_grad(Xb, W, Y):
    P = softmax(Xb @ W)
    return Xb.T @ (P - Y) / Xb.shape[0]


def numerical_grad(Xb, W, Y, eps=1e-6):
    G = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += eps
            Wm[i, j] -= eps
            G[i, j] = (cross_entropy(Xb, Wp, Y) - cross_entropy(Xb, Wm, Y)) / (2 * eps)
    return G


def train(Xb, Y, lr=0.5, n_iter=2000, l2=0.0):
    W = np.zeros((Xb.shape[1], K))
    losses = []
    for _ in range(n_iter):
        losses.append(cross_entropy(Xb, W, Y))
        W -= lr * (analytical_grad(Xb, W, Y) + l2 * W)
    losses.append(cross_entropy(Xb, W, Y))
    return W, np.array(losses)


def predict_proba(Xb, W):
    return softmax(Xb @ W)


def predict(Xb, W):
    return np.argmax(softmax(Xb @ W), axis=1)


Ytr = onehot(y_tr)
Xtr_b, Xte_b = add_bias(Xtr), add_bias(Xte)

# 梯度校验：解析梯度 vs 数值梯度
W0 = np.random.RandomState(0).randn(Xtr_b.shape[1], K) * 0.1
ga = analytical_grad(Xtr_b, W0, Ytr)
gn = numerical_grad(Xtr_b, W0, Ytr)
grad_max_diff = float(np.max(np.abs(ga - gn)))
grad_mean_ratio = float(np.mean(np.abs(ga - gn) / (np.abs(gn) + 1e-9)))

# 训练
LR, N_ITER = 0.5, 2000
W_hat, losses = train(Xtr_b, Ytr, lr=LR, n_iter=N_ITER)

pred_te = predict(Xte_b, W_hat)
prob_te = predict_proba(Xte_b, W_hat)
acc_hand = float(accuracy_score(y_te, pred_te))

# 与 sklearn 对照，验证手写实现正确
sk = LogisticRegression(max_iter=1000, random_state=RNG).fit(Xtr, y_tr)
acc_sklearn = float(accuracy_score(y_te, sk.predict(Xte)))

cm = confusion_matrix(y_te, pred_te)
prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred_te, labels=[0, 1, 2])

# 挑一个"边界上"的样本：top1 与 top2 概率差最小
srt = np.sort(prob_te, axis=1)
margin = srt[:, -1] - srt[:, -2]
b_idx = int(np.argmin(margin))
b_probs = prob_te[b_idx].tolist()
b_true, b_pred = int(y_te[b_idx]), int(pred_te[b_idx])
b_margin = float(margin[b_idx])

# ------------------------------------------------------------ 图1 三类散点
f, ax = plt.subplots(figsize=(6.4, 4.8))
for k, (col, name) in enumerate(zip([C["orange"], C["sky"], C["green"]], CN)):
    m = y == k
    ax.scatter(X_all[m, 2], X_all[m, 3], s=42, color=col, label=name,
               alpha=.85, edgecolor="white", linewidth=.6)
ax.set_xlabel("花瓣长度 (cm)")
ax.set_ylabel("花瓣宽度 (cm)")
ax.set_title("鸢尾花三类分布：山鸢尾分得开，另两类挤在一起")
ax.legend(frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
f.tight_layout()
f.savefig(os.path.join(FIG, "fig1_iris_overlap.png"))
plt.close(f)

# --------------------------------------------------- 图2 softmax 概率输出
f, ax = plt.subplots(figsize=(6.4, 4.2))
cols = [C["orange"], C["sky"], C["green"]]
bars = ax.bar(CN, b_probs, color=cols, edgecolor="white")
for b, p in zip(bars, b_probs):
    ax.text(b.get_x() + b.get_width() / 2, p + .02, f"{p:.3f}",
            ha="center", fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("概率")
ax.set_title(f"边界样本的三类概率：差距只有 {b_margin:.4f}")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
f.tight_layout()
f.savefig(os.path.join(FIG, "fig2_softmax_proba.png"))
plt.close(f)

# ------------------------------------------------------- 图3 训练损失曲线
f, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(np.arange(len(losses)), losses, color=C["verm"], lw=2)
ax.set_xlabel(f"迭代次数（学习率 {LR}）")
ax.set_ylabel("交叉熵损失")
ax.set_title(f"损失从 {losses[0]:.4f} 降到 {losses[-1]:.4f}")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
f.tight_layout()
f.savefig(os.path.join(FIG, "fig3_train_loss.png"))
plt.close(f)

# ------------------------------------------------------- 图4 梯度校验
f, ax = plt.subplots(figsize=(5.6, 5.2))
flat_a, flat_n = ga.ravel(), gn.ravel()
lim = max(np.abs(flat_a).max(), np.abs(flat_n).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], color=C["grey"], ls="--", lw=1, label="完全一致的参考线")
ax.scatter(flat_a, flat_n, s=46, color=C["blue"], alpha=.8, edgecolor="white")
ax.set_xlabel("解析梯度（推导出来的公式）")
ax.set_ylabel("数值梯度（扰动算出来的）")
ax.set_title(f"梯度校验：最大偏差 {grad_max_diff:.2e}")
ax.legend(frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
f.tight_layout()
f.savefig(os.path.join(FIG, "fig4_grad_check.png"))
plt.close(f)


# --------------------------------------------- 决策边界（花瓣两特征，线性）
def poly2(A, degree):
    cols = [np.ones(len(A))]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append((A[:, 0] ** i) * (A[:, 1] ** (d - i)))
    return np.column_stack(cols)


def build(idx, degree, l2=0.0):
    A = X_all[:, idx]
    Atr, Ate, btr, bte = train_test_split(A, y, test_size=0.30, stratify=y, random_state=RNG)
    sc = StandardScaler().fit(Atr)
    Atr_s, Ate_s = sc.transform(Atr), sc.transform(Ate)
    if degree == 1:
        Ptr, Pte = add_bias(Atr_s), add_bias(Ate_s)
    else:
        Ptr, Pte = poly2(Atr_s, degree), poly2(Ate_s, degree)
    W, _ = train(Ptr, onehot(btr), lr=0.5, n_iter=3000, l2=l2)
    acc_tr = float(accuracy_score(btr, predict(Ptr, W)))
    acc_te = float(accuracy_score(bte, predict(Pte, W)))
    return dict(sc=sc, W=W, acc_tr=acc_tr, acc_te=acc_te, degree=degree, idx=idx,
                Atr=Atr, Ate=Ate, btr=btr, bte=bte)


def draw_boundary(ax, m, title, xlabel, ylabel, show_pts=True):
    A = X_all[:, m["idx"]]
    x_min, x_max = A[:, 0].min() - .4, A[:, 0].max() + .4
    y_min, y_max = A[:, 1].min() - .4, A[:, 1].max() + .4
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 320),
                         np.linspace(y_min, y_max, 320))
    grid = np.c_[gx.ravel(), gy.ravel()]
    gs = m["sc"].transform(grid)
    if m["degree"] == 1:
        G = add_bias(gs)
    else:
        G = poly2(gs, m["degree"])
    Z = predict(G, m["W"]).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-.5, .5, 1.5, 2.5],
                colors=[C["orange"], C["sky"], C["green"]], alpha=.28)
    ax.contour(gx, gy, Z, levels=[.5, 1.5], colors=[C["black"]], linewidths=1.2, alpha=.75)
    if show_pts:
        for k, col in enumerate([C["orange"], C["sky"], C["green"]]):
            mk = y == k
            ax.scatter(A[mk, 0], A[mk, 1], s=26, color=col, edgecolor="white",
                       linewidth=.5, alpha=.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 花瓣（容易分）线性边界
m_petal = build([2, 3], degree=1)
f, ax = plt.subplots(figsize=(6.4, 5.2))
draw_boundary(ax, m_petal, f"花瓣特征的线性决策边界（测试集准确率 {m_petal['acc_te']:.3f}）",
              "花瓣长度 (cm)", "花瓣宽度 (cm)")
f.tight_layout()
f.savefig(os.path.join(FIG, "fig5_linear_boundary.png"))
plt.close(f)

# 花瓣特征：线性 vs 三次多项式（非线性边界）
m_sep_lin = build([0, 1], degree=1)          # 花萼线性（留作对照数字）
m_sep_poly = build([0, 1], degree=3)         # 花萼三次多项式（过拟合反例）
m_poly3 = build([2, 3], degree=3)            # 花瓣三次多项式
m_poly4 = build([2, 3], degree=4, l2=0.01)   # 花瓣四次 + L2 正则
f, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
draw_boundary(axes[0], m_petal,
              f"直线边界（测试集准确率 {m_petal['acc_te']:.3f}）", "花瓣长度 (cm)", "花瓣宽度 (cm)")
draw_boundary(axes[1], m_poly3,
              f"三次多项式特征后的弯边界（准确率 {m_poly3['acc_te']:.3f}）",
              "花瓣长度 (cm)", "花瓣宽度 (cm)")
f.tight_layout()
f.savefig(os.path.join(FIG, "fig6_nonlinear_boundary.png"))
plt.close(f)

# --------------------------------------------------------- 图7 混淆矩阵
f, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(CN, rotation=20, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(CN)
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
ax.set_xlabel("模型预测")
ax.set_ylabel("真实类别")
ax.set_title(f"三分类混淆矩阵（测试集 {N_TE} 条，准确率 {acc_hand:.3f}）")
f.tight_layout()
f.savefig(os.path.join(FIG, "fig7_confusion.png"))
plt.close(f)

# ------------------------------------------------------------------ 落盘
stats = {
    "dataset": {
        "name": "Iris",
        "source": "https://archive.ics.uci.edu/dataset/53/iris （sklearn.datasets.load_iris 内置）",
        "n_total": int(X_all.shape[0]),
        "n_features": int(X_all.shape[1]),
        "n_classes": K,
        "class_names": TARGETS,
        "class_names_cn": CN,
        "n_per_class": [int((y == k).sum()) for k in range(K)],
    },
    "split": {
        "test_size": 0.30, "random_state": RNG, "stratify": True,
        "n_train": int(N_TR), "n_test": int(N_TE),
    },
    "grad_check": {
        "max_abs_diff": grad_max_diff,
        "mean_rel_diff": grad_mean_ratio,
        "pass": bool(grad_max_diff < 1e-6),
    },
    "training": {
        "lr": LR, "n_iter": N_ITER,
        "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
        "loss_drop": float(losses[0] - losses[-1]),
    },
    "handwritten_vs_sklearn": {
        "handwritten_test_acc": acc_hand,
        "sklearn_test_acc": acc_sklearn,
        "acc_gap": round(abs(acc_hand - acc_sklearn), 4),
    },
    "confusion_matrix": cm.tolist(),
    "per_class": {
        TARGETS[k]: {"precision": round(float(prec[k]), 4),
                     "recall": round(float(rec[k]), 4),
                     "f1": round(float(f1[k]), 4)} for k in range(K)
    },
    "boundary_sample": {
        "test_index": b_idx, "probs": [round(p, 4) for p in b_probs],
        "true_class": TARGETS[b_true], "pred_class": TARGETS[b_pred],
        "top1_top2_margin": round(b_margin, 4),
    },
    "decision_boundary": {
        "petal_linear_test_acc": round(m_petal["acc_te"], 4),
        "petal_poly3_test_acc": round(m_poly3["acc_te"], 4),
        "petal_poly4_l2_test_acc": round(m_poly4["acc_te"], 4),
        "petal_improvement_poly3": round(m_poly3["acc_te"] - m_petal["acc_te"], 4),
        "sepal_linear_test_acc": round(m_sep_lin["acc_te"], 4),
        "sepal_poly3_test_acc": round(m_sep_poly["acc_te"], 4),
        "sepal_improvement_poly3": round(m_sep_poly["acc_te"] - m_sep_lin["acc_te"], 4),
    },
}

with open(os.path.join(ROOT, "stats.json"), "w", encoding="utf-8") as fp:
    json.dump(stats, fp, ensure_ascii=False, indent=2)

print(json.dumps(stats, ensure_ascii=False, indent=2))
print("\nfigures ->", FIG)
