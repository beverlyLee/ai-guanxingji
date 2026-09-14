# -*- coding: utf-8 -*-
"""
M12 实验脚本：支持向量机（SVM）在 UCI spambase 诈骗短信数据集上的真实实验。
所有数字写入 ../stats.json，供 figures.py 与正文引用。RNG=42，可复现。
运行：python code/experiment.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "_tmp")
os.makedirs(TMP, exist_ok=True)
os.environ["TMPDIR"] = TMP
RNG = 42

# ---------- 1. 加载数据 ----------
d = fetch_openml(name="spambase", version=1, as_frame="auto", parser="auto")
X_raw = d.data  # DataFrame 4601 x 57
y = d.target.astype(int).to_numpy()  # 0=正常邮件(ham) 1=诈骗邮件(spam)
feature_names = list(X_raw.columns)
n_samples, n_features = X_raw.shape
n_spam = int((y == 1).sum())
n_ham = int((y == 0).sum())

# 57 个特征的三类来源（讲清"数据从哪来"）
word_feats = [c for c in feature_names if c.startswith("word_freq_")]
char_feats = [c for c in feature_names if c.startswith("char_freq_")]
cap_feats = [c for c in feature_names if c.startswith("capital_run_length")]

print(f"样本 {n_samples} 特征 {n_features} 正常 {n_ham} 诈骗 {n_spam}")
print(f"特征构成: 词频 {len(word_feats)} + 字符频 {len(char_feats)} + 大写连续长度 {len(cap_feats)}")

# 每个词/字符特征在诈骗邮件 vs 正常邮件里的平均出现频率，演示"数据从哪来"
spam_mean = X_raw[y == 1].mean()
ham_mean = X_raw[y == 0].mean()
diff = (spam_mean - ham_mean).sort_values(ascending=False)
top_spam_words = [
    {"feature": k, "spam_mean": round(float(spam_mean[k]), 4),
     "ham_mean": round(float(ham_mean[k]), 4), "gap": round(float(diff[k]), 4)}
    for k in diff.index[:12]
]
# 反向：正常邮件里更常见的词（验证区分度双向）
top_ham_words = [
    {"feature": k, "spam_mean": round(float(spam_mean[k]), 4),
     "ham_mean": round(float(ham_mean[k]), 4), "gap": round(float(diff[k]), 4)}
    for k in diff.index[::-1][:6]
]

# ---------- 2. 标准化（SVM 对尺度敏感） ----------
scaler = StandardScaler()
Xs = scaler.fit_transform(X_raw)

# ---------- 3. 线性 SVM：最大间隔与支持向量 ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)


def svm_margin(model):
    """最大间隔宽度 = 2 / ||w||，w 为线性核的法向量。"""
    w = model.coef_[0]
    return float(2.0 / np.linalg.norm(w))


base = SVC(kernel="linear", C=1.0, random_state=RNG)
base.fit(Xs, y)
base_train = float(base.score(Xs, y))
base_cv = cross_val_score(base, Xs, y, cv=skf, scoring="accuracy")
base_cv_mean = float(base_cv.mean())
base_cv_std = float(base_cv.std())
base_margin = svm_margin(base)
base_nsv = int(base.n_support_.sum())
base_nsv_ratio = base_nsv / n_samples

print(f"线性SVM C=1: 训练 {base_train:.3f} CV {base_cv_mean:.3f} 间隔 {base_margin:.4f} 支持向量 {base_nsv}")

# ---------- 4. 软间隔 C 参数扫描 ----------
c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
c_records = []
for C in c_grid:
    m = SVC(kernel="linear", C=C, random_state=RNG)
    m.fit(Xs, y)
    tr = float(m.score(Xs, y))
    cv = float(cross_val_score(m, Xs, y, cv=skf, scoring="accuracy").mean())
    nsv = int(m.n_support_.sum())
    marg = svm_margin(m)
    c_records.append({
        "C": C,
        "train_acc": round(tr, 4),
        "cv_acc": round(cv, 4),
        "n_support": nsv,
        "n_support_ratio": round(nsv / n_samples, 4),
        "margin_width": round(marg, 4),
    })
    print(f"  C={C:<6} train {tr:.3f} cv {cv:.3f} SV {nsv} 间隔 {marg:.4f}")

# ---------- 5. 核函数对比 ----------
kernel_grid = ["linear", "rbf", "poly"]
kernel_records = []
for k in kernel_grid:
    m = SVC(kernel=k, random_state=RNG)
    cvs = cross_val_score(m, Xs, y, cv=skf, scoring="accuracy")
    m.fit(Xs, y)
    nsv = int(m.n_support_.sum())
    kernel_records.append({
        "kernel": k,
        "cv_mean": round(float(cvs.mean()), 4),
        "cv_std": round(float(cvs.std()), 4),
        "cv_mean_pp": round(float(cvs.mean()) * 100, 1),
        "cv_std_pp": round(float(cvs.std()) * 100, 1),
        "n_support": nsv,
    })
    print(f"核={k:<6} CV {float(cvs.mean()):.3f}±{float(cvs.std()):.3f} SV {nsv}")

# 在 2 个主成分上各核的决策边界（演示核技巧映射效果，供配图）
pca = PCA(n_components=2, random_state=RNG)
X2 = pca.fit_transform(Xs)
explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
kernel_2d = {}
for k in kernel_grid:
    m = SVC(kernel=k, random_state=RNG)
    m.fit(X2, y)
    cvs = cross_val_score(m, X2, y, cv=skf, scoring="accuracy")
    kernel_2d[k] = {
        "cv_mean_pp": round(float(cvs.mean()) * 100, 1),
        "cv_std_pp": round(float(cvs.std()) * 100, 1),
    }
    print(f"  2D核={k:<6} CV(2维) {float(cvs.mean()):.3f}")

# ---------- 6. PCA 2D 决策边界（线性核，C=1） ----------
svm2 = SVC(kernel="linear", C=1.0, random_state=RNG)
svm2.fit(X2, y)
sv_mask = np.zeros(n_samples, dtype=bool)
sv_mask[svm2.support_] = True
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
)
Z = svm2.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ---------- 保存 stats.json ----------
stats = {
    "dataset": "UCI Spambase (OpenML spambase v1)",
    "task": "诈骗短信识别(ham=0 / spam=1)",
    "n_samples": n_samples,
    "n_features": n_features,
    "n_ham": n_ham,
    "n_spam": n_spam,
    "spam_rate": round(n_spam / n_samples, 4),
    "feature_groups": {
        "word_freq_count": len(word_feats),
        "char_freq_count": len(char_feats),
        "capital_run_length_count": len(cap_feats),
        "word_feats_sample": word_feats[:12],
        "char_feats": char_feats,
        "cap_feats": cap_feats,
    },
    "top_spam_words": top_spam_words,
    "top_ham_words": top_ham_words,
    "base_linear_svm": {
        "C": 1.0,
        "train_acc": round(base_train, 4),
        "cv_mean": round(base_cv_mean, 4),
        "cv_std": round(base_cv_std, 4),
        "cv_mean_pp": round(base_cv_mean * 100, 1),
        "margin_width": round(base_margin, 4),
        "n_support": base_nsv,
        "n_support_ratio": round(base_nsv_ratio, 4),
    },
    "c_sweep": c_records,
    "kernel_compare": kernel_records,
    "pca": {
        "explained_variance_ratio": explained,
        "explained_sum_2d": round(float(sum(explained)), 4),
        "kernel_2d": kernel_2d,
    },
    "rng": RNG,
}

with open(os.path.join(ROOT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# 保存处理好的数据，供配图复用
np.save(os.path.join(ROOT, "data", "X_raw.npy"), X_raw.to_numpy())
np.save(os.path.join(ROOT, "data", "X_scaled.npy"), Xs)
np.save(os.path.join(ROOT, "data", "y.npy"), y)
with open(os.path.join(ROOT, "data", "feat_names.json"), "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False)
X_raw.assign(target=y).to_csv(os.path.join(ROOT, "data", "processed.csv"), index=False)

# PCA 2D 决策边界数据
np.save(os.path.join(ROOT, "data", "pca_X.npy"), X2)
np.save(os.path.join(ROOT, "data", "pca_sv.npy"), sv_mask)
np.save(os.path.join(ROOT, "data", "pca_xx.npy"), xx)
np.save(os.path.join(ROOT, "data", "pca_yy.npy"), yy)
np.save(os.path.join(ROOT, "data", "pca_Z.npy"), Z)

print("stats.json 已写入")
print("C=1 间隔宽度", round(base_margin, 4), "支持向量数", base_nsv, "占比", round(base_nsv_ratio, 4))
print("最优核 CV:", max(kernel_records, key=lambda r: r["cv_mean"])["kernel"],
      round(max(r["cv_mean"] for r in kernel_records) * 100, 1), "%")
