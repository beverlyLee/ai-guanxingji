"""
M7 实验脚本：用真实公开数据集演示「模型评估方法」整条链路
数据集：UCI Default of Credit Card Clients（信用卡违约预测，30000 条，正类占 22.12%）
知识点：sklearn / 数据集切分 / 交叉验证 / 混淆矩阵 / 评估指标对比 / 阈值影响 / ROC 曲线

所有数字写入 ../stats.json，配图写入 ../figures/，保证正文/脚本/配图三处一致。
运行：python experiment.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, roc_curve)

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
DATA = os.path.join(HERE, "..", "data", "default_cc.xls")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

stats = {}

# ---------- 1. 载入数据 ----------
df = pd.read_excel(DATA, header=1)
df = df.drop(columns=["ID"])
y = df["default payment next month"].astype(int).values
X = df.drop(columns=["default payment next month"]).values
n_total, n_feat = X.shape
pos = int(y.sum())
stats["dataset"] = {
    "name": "UCI Default of Credit Card Clients",
    "n_total": int(n_total),
    "n_features": int(n_feat),
    "n_positive": pos,
    "n_negative": int(n_total - pos),
    "positive_rate": round(pos / n_total, 4),
    "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
}

# ---------- 2. 数据集切分 ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RNG)
stats["split"] = {
    "test_size": 0.30, "random_state": RNG, "stratify": True,
    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "positive_rate_train": round(float(y_tr.mean()), 4),
    "positive_rate_test": round(float(y_te.mean()), 4),
}

# ---------- 3. 训练逻辑回归（带标准化，构成 sklearn pipeline）----------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=RNG)),
])
pipe.fit(X_tr, y_tr)
prob_te = pipe.predict_proba(X_te)[:, 1]
pred_te = (prob_te >= 0.5).astype(int)

# ---------- 4. 混淆矩阵 + 评估指标 ----------
tn, fp, fn, tp = confusion_matrix(y_te, pred_te).ravel()
acc = accuracy_score(y_te, pred_te)
prec = precision_score(y_te, pred_te)
rec = recall_score(y_te, pred_te)
f1 = f1_score(y_te, pred_te)
auc = roc_auc_score(y_te, prob_te)

# 多数类基线（不学任何东西，永远预测「不违约」）
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
acc_dummy = accuracy_score(y_te, dummy.predict(X_te))

stats["metrics_default_threshold"] = {
    "threshold": 0.5,
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4),
    "auc": round(float(auc), 4),
    "majority_baseline_accuracy": round(float(acc_dummy), 4),
}

# ---------- 5. 交叉验证（StratifiedKFold，不平衡数据不能用普通 K 折）----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
cv = cross_val_score(pipe, X, y, cv=skf, scoring="recall")
cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
stats["cross_validation"] = {
    "n_splits": 5, "shuffle": True, "random_state": RNG,
    "scoring": "recall",
    "fold_scores": [round(float(s), 4) for s in cv],
    "recall_mean": round(float(cv.mean()), 4),
    "recall_std": round(float(cv.std()), 4),
    "accuracy_mean": round(float(cv_acc.mean()), 4),
    "accuracy_std": round(float(cv_acc.std()), 4),
}

# ---------- 6. 阈值影响 ----------
prec_c, rec_c, thr = precision_recall_curve(y_te, prob_te)
# 选一个更低的阈值，看召回率如何上升、精确率如何下降
thr_low = 0.30
pred_low = (prob_te >= thr_low).astype(int)
tn_l, fp_l, fn_l, tp_l = confusion_matrix(y_te, pred_low).ravel()
stats["threshold_sweep"] = {
    "default_threshold": 0.5,
    "low_threshold": thr_low,
    "default": {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                "TP": int(tp), "FN": int(fn), "FP": int(fp)},
    "low": {"precision": round(float(precision_score(y_te, pred_low)), 4),
            "recall": round(float(recall_score(y_te, pred_low)), 4),
            "TP": int(tp_l), "FN": int(fn_l), "FP": int(fp_l)},
}

# ---------- 7. ROC 曲线 ----------
fpr, tpr, _ = roc_curve(y_te, prob_te)
stats["roc"] = {"auc": round(float(auc), 4),
                "n_points": int(len(fpr))}

with open(os.path.join(HERE, "..", "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# ================= 配图 =================
# 图1：类别不平衡（Q1 引子）
fig, ax = plt.subplots(figsize=(6, 3.2))
bars = ax.barh(["未违约", "违约"], [n_total - pos, pos],
               color=[C["grey"], C["verm"]])
for b, v in zip(bars, [n_total - pos, pos]):
    ax.text(b.get_width() + 300, b.get_y() + b.get_height() / 2,
            f"{v:,}（{v / n_total:.1%}）", va="center", fontsize=10)
ax.set_xlim(0, n_total * 1.15)
ax.set_title("数据集里，违约客户只占 22.12%", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_imbalance.png")); plt.close(fig)

# 图2：混淆矩阵（Q1）
fig, ax = plt.subplots(figsize=(4.6, 4.0))
cm = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm, cmap="Oranges")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["预测未违约", "预测违约"])
ax.set_yticklabels(["实际未违约", "实际违约"])
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                fontsize=12, color="black")
ax.set_title("混淆矩阵：模型到底错在哪", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_confusion.png")); plt.close(fig)

# 图3：评估指标对比（Q2）
fig, ax = plt.subplots(figsize=(6.5, 3.6))
names = ["accuracy", "precision", "recall", "F1", "AUC"]
vals = [acc, prec, rec, f1, auc]
cols = [C["orange"], C["sky"], C["green"], C["blue"], C["purple"]]
b = ax.bar(names, vals, color=cols)
for rect, v in zip(b, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", fontsize=10)
ax.axhline(acc_dummy, ls="--", color=C["grey"], lw=1)
ax.text(4.3, acc_dummy + 0.01, f"多数类基线 {acc_dummy:.3f}", fontsize=9, color=C["grey"])
ax.set_ylim(0, 1.05)
ax.set_title("五个指标摆一起，accuracy 的误导一目了然", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_metrics.png")); plt.close(fig)

# 图4：交叉验证（Q3）
fig, ax = plt.subplots(figsize=(6, 3.4))
fold_idx = np.arange(1, 6)
ax.plot(fold_idx, cv, "o-", color=C["green"], lw=2, label="各折 recall")
ax.axhline(cv.mean(), ls="--", color=C["verm"], lw=1.2, label=f"均值 {cv.mean():.3f}")
ax.fill_between(fold_idx, cv.mean() - cv.std(), cv.mean() + cv.std(),
               color=C["green"], alpha=0.15, label=f"±1 标准差 {cv.std():.3f}")
ax.set_xticks(fold_idx); ax.set_ylim(0, 1)
ax.set_xlabel("折（Fold）"); ax.set_ylabel("recall")
ax.set_title("5 折交叉验证：这次准，换批数据还准吗", fontsize=13)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_cv.png")); plt.close(fig)

# 图5：阈值影响（Q4）
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(thr, prec_c[:-1], color=C["blue"], lw=2, label="精确率 precision")
ax.plot(thr, rec_c[:-1], color=C["verm"], lw=2, label="召回率 recall")
ax.axvline(0.5, ls="--", color=C["grey"], lw=1)
ax.text(0.5, 0.05, "默认 0.5", fontsize=9, color=C["grey"], ha="center")
ax.axvline(thr_low, ls="--", color=C["green"], lw=1)
ax.text(thr_low, 0.05, f"调到 {thr_low}", fontsize=9, color=C["green"], ha="center")
ax.set_xlabel("决策阈值"); ax.set_ylabel("指标值")
ax.set_title("阈值拧一拧：召回率和精确率此消彼长", fontsize=13)
ax.legend(fontsize=9, frameon=False, loc="center right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_threshold.png")); plt.close(fig)

# 图6：ROC 曲线（Q5）
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.plot(fpr, tpr, color=C["purple"], lw=2, label=f"ROC（AUC={auc:.3f}）")
ax.plot([0, 1], [0, 1], ls="--", color=C["grey"], lw=1, label="随机猜")
ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
ax.set_title("一条 ROC 曲线，看完模型全局表现", fontsize=13)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_roc.png")); plt.close(fig)

print("DONE. stats.json + 6 figures written.")
print(json.dumps(stats, ensure_ascii=False, indent=2))
