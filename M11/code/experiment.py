# -*- coding: utf-8 -*-
"""
M11 实验脚本：随机森林 / 提升 / Stacking 在克利夫兰心脏病数据集上的真实对比。
所有数字写入 ../stats.json，供 figures.py 与正文引用。RNG=42，可复现。
运行：python code/experiment.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "_tmp")
os.makedirs(TMP, exist_ok=True)
os.environ["TMPDIR"] = TMP
RNG = 42

# ---------- 1. 加载数据 ----------
d = fetch_openml("cleveland", version=1, as_frame="auto", parser="auto")
X_raw = d.data
y = (d.target > 0).astype(int).to_numpy()
feature_names = list(X_raw.columns)
n_missing = int(X_raw.isna().sum().sum())
missing_per_col = {c: int(X_raw[c].isna().sum()) for c in feature_names}

imp = SimpleImputer(strategy="most_frequent")
X = imp.fit_transform(X_raw)
X = pd.DataFrame(X, columns=feature_names)
n_samples, n_features = X.shape
pos_rate = float(y.mean())

print(f"样本 {n_samples} 特征 {n_features} 缺失 {n_missing} 患病比例 {pos_rate:.3f}")

# ---------- 2. 单棵决策树：过拟合与方差演示 ----------
depth_grid = [1, 2, 3, 4, 5, 6, 8, 10, 12, None]
train_acc, cv_acc, cv_std = [], [], []
for md in depth_grid:
    tree = DecisionTreeClassifier(max_depth=md, random_state=RNG)
    tree.fit(X, y)
    train_acc.append(float(tree.score(X, y)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    s = cross_val_score(tree, X, y, cv=skf, scoring="accuracy")
    cv_acc.append(float(s.mean()))
    cv_std.append(float(s.std()))

deep_tree = DecisionTreeClassifier(random_state=RNG)  # 不限制深度
deep_tree.fit(X, y)
deep_train = float(deep_tree.score(X, y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
deep_cv = cross_val_score(deep_tree, X, y, cv=skf, scoring="accuracy")
deep_mean, deep_std = float(deep_cv.mean()), float(deep_cv.std())

shallow = DecisionTreeClassifier(max_depth=4, random_state=RNG)
sh_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
sh_cv = cross_val_score(shallow, X, y, cv=sh_skf, scoring="accuracy")
shallow_mean, shallow_std = float(sh_cv.mean()), float(sh_cv.std())

# ---------- 3. Bagging（方差削减演示） ----------
bag = BaggingClassifier(random_state=RNG)
bag_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
bag_cv = cross_val_score(bag, X, y, cv=bag_skf, scoring="accuracy")
bag_mean, bag_std = float(bag_cv.mean()), float(bag_cv.std())

# ---------- 4. 随机森林 ----------
rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=RNG)
rf.fit(X, y)
rf_oob = float(rf.oob_score_)
rf_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
rf_cv = cross_val_score(rf, X, y, cv=rf_skf, scoring="accuracy")
rf_mean, rf_std = float(rf_cv.mean()), float(rf_cv.std())

imp_array = rf.feature_importances_
order = np.argsort(imp_array)[::-1]
feature_importances = {feature_names[i]: float(imp_array[i]) for i in order}
top_feats = [(feature_names[i], float(imp_array[i])) for i in order[:5]]

# 排列重要性（更稳健的对照）
perm = permutation_importance(rf, X, y, n_repeats=20, random_state=RNG, scoring="accuracy")
perm_imp = {feature_names[i]: float(perm.importances_mean[i]) for i in np.argsort(perm.importances_mean)[::-1]}

# RF 的 OOB 误差随树数量收敛曲线
n_est_grid = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]
oob_curve = []
for k in n_est_grid:
    r = RandomForestClassifier(n_estimators=k, oob_score=True, random_state=RNG)
    r.fit(X, y)
    oob_curve.append([k, float(1.0 - r.oob_score_)])
oob_curve = [[int(k), round(e, 4)] for k, e in oob_curve]

# ---------- 5. 提升算法 ----------
gbm = GradientBoostingClassifier(random_state=RNG)
gbm_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
gbm_cv = cross_val_score(gbm, X, y, cv=gbm_skf, scoring="accuracy")
gbm_mean, gbm_std = float(gbm_cv.mean()), float(gbm_cv.std())

ada = AdaBoostClassifier(random_state=RNG)
ada_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
ada_cv = cross_val_score(ada, X, y, cv=ada_skf, scoring="accuracy")
ada_mean, ada_std = float(ada_cv.mean()), float(ada_cv.std())

# ---------- 6. Stacking ----------
stack = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, random_state=RNG)),
        ("gbm", GradientBoostingClassifier(random_state=RNG)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
)
stack_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
stack_cv = cross_val_score(stack, X, y, cv=stack_skf, scoring="accuracy")
stack_mean, stack_std = float(stack_cv.mean()), float(stack_cv.std())

# ---------- 7. 模型对比汇总 ----------
model_comparison = [
    {"model": "单棵浅树(max_depth=4)", "mean": shallow_mean, "std": shallow_std},
    {"model": "单棵深树(不剪枝)", "mean": deep_mean, "std": deep_std},
    {"model": "Bagging(深树)", "mean": bag_mean, "std": bag_std},
    {"model": "随机森林", "mean": rf_mean, "std": rf_std},
    {"model": "AdaBoost", "mean": ada_mean, "std": ada_std},
    {"model": "GradientBoosting", "mean": gbm_mean, "std": gbm_std},
    {"model": "Stacking", "mean": stack_mean, "std": stack_std},
]
for r in model_comparison:
    r["mean"] = round(r["mean"], 3)
    r["std"] = round(r["std"], 3)
    r["mean_pp"] = round(r["mean"] * 100, 1)
    r["std_pp"] = round(r["std"] * 100, 1)

overfit_gap_pp = round((1.0 - deep_mean) * 100, 1)

# ---------- 保存 ----------
stats = {
    "dataset": "UCI Cleveland (OpenML cleveland v1)",
    "n_samples": n_samples,
    "n_features": n_features,
    "n_missing": n_missing,
    "missing_per_col": missing_per_col,
    "pos_rate": round(pos_rate, 3),
    "feature_names": feature_names,
    "depth_sweep": {
        "depth": [("None" if d is None else d) for d in depth_grid],
        "train_acc": [round(a, 3) for a in train_acc],
        "cv_acc": [round(a, 3) for a in cv_acc],
        "cv_std": [round(a, 3) for a in cv_std],
    },
    "deep_tree": {"train_acc": round(deep_train, 3), "cv_mean": round(deep_mean, 3), "cv_std": round(deep_std, 3)},
    "shallow_tree": {"cv_mean": round(shallow_mean, 3), "cv_std": round(shallow_std, 3)},
    "bagging": {"cv_mean": round(bag_mean, 3), "cv_std": round(bag_std, 3)},
    "random_forest": {
        "oob_score": round(rf_oob, 3),
        "cv_mean": round(rf_mean, 3),
        "cv_std": round(rf_std, 3),
        "n_estimators": 300,
        "feature_importances": feature_importances,
        "top5": top_feats,
    },
    "permutation_importance": perm_imp,
    "rf_oob_curve": oob_curve,
    "adaboost": {"cv_mean": round(ada_mean, 3), "cv_std": round(ada_std, 3)},
    "gradient_boosting": {"cv_mean": round(gbm_mean, 3), "cv_std": round(gbm_std, 3)},
    "stacking": {"cv_mean": round(stack_mean, 3), "cv_std": round(stack_std, 3)},
    "model_comparison": model_comparison,
    "model_comparison_pp": [{"model": r["model"], "mean_pp": r["mean_pp"], "std_pp": r["std_pp"]} for r in model_comparison],
    "overfit_gap_pp": overfit_gap_pp,
    "theory": {"bootstrap_in_bag_rate": 0.632, "bootstrap_oob_rate": 0.368},
    "rng": RNG,
}

with open(os.path.join(ROOT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# 保存处理好的数据，供配图复用
np.save(os.path.join(ROOT, "data", "X.npy"), X.to_numpy())
np.save(os.path.join(ROOT, "data", "y.npy"), y)
with open(os.path.join(ROOT, "data", "feat_names.json"), "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False)
X.assign(target=y).to_csv(os.path.join(ROOT, "data", "processed.csv"), index=False)

print("stats.json 已写入")
print("OOB", round(rf_oob, 3), "top3", top_feats[:3])
print("deep tree cv", round(deep_mean, 3), "std", round(deep_std, 3))
print("bagging cv", round(bag_mean, 3), "std", round(bag_std, 3))
print("stacking cv", round(stack_mean, 3))
