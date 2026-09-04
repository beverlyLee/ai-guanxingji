# -*- coding: utf-8 -*-
"""
M1 机器学习导论：写规则 vs 从数据学（真实对照实验）+ 概念嵌套图 + AI 时间线
数据集：sklearn 内置 load_breast_cancer（威斯康星乳腺癌诊断集，569 例 x 30 特征，零下载）
所有数字真实运行结果，落盘 ../stats.json 与 ../figures/*.png
"""
import json, os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Okabe-Ito 色盲安全配色 ----
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
     "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00", "purp": "#CC79A7"}
plt.rcParams.update({
    "font.family": "PingFang SC",
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# =====================================================================
# 1. 数据：乳腺癌诊断集。良恶性二分类，30 个细胞核形态特征
# =====================================================================
data = load_breast_cancer()
X, y = data.data, data.target          # 1=良性 0=恶性
feat_names = list(data.feature_names)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
N_TOTAL, N_TR, N_TE, N_FEAT = X.shape[0], Xtr.shape[0], Xte.shape[0], X.shape[1]

# =====================================================================
# 2. 实验一：人写规则（只盯 1 个特征）
#    规则只允许在训练集上"划线"，不碰测试集标签——这是人肉调规则的诚实做法
# =====================================================================
# 特征 0 = mean radius（肿瘤平均半径）。医学直觉：半径越大越像恶性。
i_feat = 0
med_bad = np.median(Xtr[ytr == 0, i_feat])   # 恶性组训练集中位数
med_good = np.median(Xtr[ytr == 1, i_feat])  # 良性组训练集中位数
thr = (med_bad + med_good) / 2               # 人肉规则：半径超过这个值判恶性
rule_pred = np.where(Xte[:, i_feat] > thr, 0, 1)
rule_acc = accuracy_score(yte, rule_pred)

# =====================================================================
# 3. 实验二：机器学习（同一个特征，让机器自己找规律）
# =====================================================================
knn1 = KNeighborsClassifier(5).fit(Xtr[:, [i_feat]], ytr)
knn1_acc = accuracy_score(yte, knn1.predict(Xte[:, [i_feat]]))

# =====================================================================
# 4. 实验三：机器学习 + 全部 30 个特征
# =====================================================================
sc = StandardScaler().fit(Xtr)
knn30 = KNeighborsClassifier(5).fit(sc.transform(Xtr), ytr)
knn30_acc = accuracy_score(yte, knn30.predict(sc.transform(Xte)))
log30 = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
log30_acc = accuracy_score(yte, log30.predict(sc.transform(Xte)))

print(f"数据：{N_TOTAL} 例 x {N_FEAT} 特征（训练 {N_TR} / 测试 {N_TE}）")
print(f"人写规则（只看平均半径, 阈值 {thr:.2f}）: {rule_acc:.4f}")
print(f"机器学习（同一特征, KNN k=5）      : {knn1_acc:.4f}")
print(f"机器学习（30 特征, KNN k=5）       : {knn30_acc:.4f}")
print(f"机器学习（30 特征, 逻辑回归）      : {log30_acc:.4f}")

# =====================================================================
# 5. fig1：AI > 机器学习 > 深度学习（> 大模型）嵌套关系
# =====================================================================
fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
boxes = [
    (0.3, 0.3, 9.4, 6.4, C["sky"],   0.25, "AI 人工智能", "目标：让机器做需要人类智能的事\n下棋程序 · 专家系统 · 人脸识别 · 对话助手"),
    (1.2, 1.1, 7.8, 4.6, C["orange"],0.30, "机器学习", "方法：从数据里自动找规律\n房价预测 · 垃圾邮件过滤 · 信用评分"),
    (2.1, 1.9, 6.0, 2.8, C["green"], 0.35, "深度学习", "主力分支：多层神经网络\n人脸识别 · 语音识别"),
]
for (x0, y0, w, h, color, alpha, title, sub) in boxes:
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fc=color, ec=C["black"], lw=0.8, alpha=alpha, zorder=1))
    ax.text(x0 + 0.25, y0 + h - 0.42, title, fontsize=13, fontweight="bold", zorder=3)
    ax.text(x0 + 0.25, y0 + h - 1.05, sub, fontsize=9, zorder=3)
ax.add_patch(plt.Rectangle((3.0, 2.1), 4.2, 1.5, fc=C["verm"], ec=C["black"], lw=0.8, alpha=0.45, zorder=2))
ax.text(3.2, 3.28, "大模型（LLM）", fontsize=12, fontweight="bold", zorder=3)
ax.text(3.2, 2.62, "深度学习的一种\nChatGPT · 文生图", fontsize=9, zorder=3)
ax.text(5.0, 0.02, "包含关系，不是并列的三件事：AI 是目标，机器学习是方法，深度学习是其中最火的一支",
        ha="center", fontsize=9.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_nesting.png")); plt.close()

# =====================================================================
# 6. fig2：写规则 vs 从数据学（真实实验结果）
# =====================================================================
names = ["人写规则\n（只看 1 个特征）", "机器学习\n（同一特征）", "机器学习\n（30 个特征 KNN）", "机器学习\n（30 特征逻辑回归）"]
vals = [rule_acc, knn1_acc, knn30_acc, log30_acc]
cols = [C["verm"], C["orange"], C["sky"], C["green"]]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
b = ax.bar(names, vals, color=cols, width=0.6)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("测试集准确率")
ax.set_title(f"同一个判断题（{N_TE} 例测试），规则与学习的能力边界")
for rect, v in zip(b, vals):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.006, f"{v:.3f}", ha="center", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_rule_vs_learn.png")); plt.close()

# =====================================================================
# 7. fig3：AI 大事时间线（真实年份，全部可查证）
# =====================================================================
events = [
    (1950, "图灵提出\n「机器能思考吗」", 0.75),
    (1956, "达特茅斯会议\n「人工智能」定名", -0.75),
    (1997, "深蓝胜\n国际象棋冠军", 0.75),
    (2012, "AlexNet 横扫 ImageNet", -0.75),
    (2016, "AlphaGo 胜李世石", 1.55),
    (2017, "Transformer 论文发表", -1.55),
    (2022, "ChatGPT 发布，\n大模型进入日常", 0.75),
]
fig, ax = plt.subplots(figsize=(9.6, 4.2))
ax.axhline(0, color=C["black"], lw=1.2)
for yr, label, h in events:
    side = 1 if h > 0 else -1
    ax.plot(yr, 0, "o", ms=8, color=C["verm"], zorder=3)
    ax.plot([yr, yr], [0, h], color=C["black"], lw=0.8)
    ax.text(yr, h + 0.08 * side, f"{yr}\n{label}", ha="center",
            va="bottom" if side > 0 else "top", fontsize=8.5)
ax.set_xlim(1944, 2029); ax.set_ylim(-3.0, 3.0)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_title("AI 这条路走了七十多年：写规则的四十年，从数据学的三十年")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_timeline.png")); plt.close()

# =====================================================================
# 8. stats.json
# =====================================================================
stats = {
    "dataset": "sklearn load_breast_cancer（威斯康星乳腺癌诊断集）",
    "n_total": int(N_TOTAL), "n_train": int(N_TR), "n_test": int(N_TE),
    "n_features": int(N_FEAT),
    "rule_1feat": {"feature": feat_names[i_feat], "threshold": round(float(thr), 2),
                    "test_acc": round(float(rule_acc), 4)},
    "knn_1feat": {"k": 5, "test_acc": round(float(knn1_acc), 4)},
    "knn_30feat": {"k": 5, "standardized": True, "test_acc": round(float(knn30_acc), 4)},
    "logreg_30feat": {"standardized": True, "test_acc": round(float(log30_acc), 4)},
}
with open(os.path.join(ROOT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("== stats.json 已写出 ==")
