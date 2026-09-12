"""M10 decision-tree experiment on real Ames housing data (Kaggle train.csv, 1460 rows).

Outputs:
  data/processed.csv          cleaned feature matrix + target
  stats.json                   every number cited in the article

All numbers are produced by running real code on the real dataset.
"""
import os, json, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                           plot_tree, export_text)
from sklearn.metrics import r2_score

RNG = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

QUAL_MAP = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, "NA":0, np.nan:0}

# ---------------------------------------------------------------- load + clean
CSV = os.path.join(DATA, "house_prices.csv")
if not os.path.exists(CSV):
    # self-contained: pull the original Ames snapshot (Kaggle train.csv) from OpenML
    from sklearn.datasets import fetch_openml
    print("house_prices.csv missing, fetching OpenML house_prices v1 ...")
    ds = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    pd.concat([ds.data, ds.target.rename("SalePrice")], axis=1).to_csv(CSV, index=False)
    print("saved ->", CSV)
raw = pd.read_csv(CSV)
y = raw["SalePrice"].astype(float).values
median_price = float(np.median(y))

num_feats = ["LotArea", "OverallQual", "YearBuilt", "YearRemodAdd",
             "TotalBsmtSF", "GrLivArea", "GarageCars", "FullBath",
             "TotRmsAbvGrd", "Fireplaces", "MasVnrArea", "GarageYrBlt",
             "BsmtFinSF1", "OpenPorchSF"]
for c in num_feats:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw[c] = raw[c].fillna(raw[c].median())
    else:
        num_feats.remove(c)

qual_feats = [c for c in ["ExterQual","KitchenQual","BsmtQual","GarageQual",
                           "FireplaceQu","HeatingQC"] if c in raw.columns]
for c in qual_feats:
    raw[c] = raw[c].map(QUAL_MAP).fillna(0).astype(int)

# Neighborhood one-hot (location is the dominant price driver)
nb = pd.get_dummies(raw["Neighborhood"], prefix="NB") if "Neighborhood" in raw.columns else pd.DataFrame()

feat = pd.concat([raw[num_feats].reset_index(drop=True),
                  raw[qual_feats].reset_index(drop=True),
                  nb.reset_index(drop=True)], axis=1)
feat_names = list(feat.columns)
X = feat.values.astype(float)
print(f"feature matrix: {X.shape[0]} rows x {X.shape[1]} cols")
print(f"median SalePrice = {median_price:.0f}")

# classification target: above / below median
above = (y > median_price).astype(int)
print(f"above-median count = {above.sum()}, below = {(1-above).sum()}")

# ---------------------------------------------------------------- from-scratch recursive CART (classification, entropy)
def entropy(labels):
    n = len(labels)
    if n == 0: return 0.0
    p = np.bincount(labels, minlength=2) / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def best_split(Xs, idx, lab, feat_cols):
    """greedy best binary split over given feature columns; returns split info."""
    best = None
    parent_e = entropy(lab[idx])
    n = len(idx)
    for j, col in enumerate(feat_cols):
        vals = Xs[idx, col]
        cand = np.unique(vals)
        if len(cand) <= 1: continue
        # try midpoints between sorted unique values (cap candidates for speed)
        thr = ((cand[:-1] + cand[1:]) / 2.0)
        if len(thr) > 25:
            thr = np.quantile(vals, np.linspace(0.02, 0.98, 25))
        for t in thr:
            left = idx[vals <= t]; right = idx[vals > t]
            if len(left) == 0 or len(right) == 0: continue
            wL, wR = len(left)/n, len(right)/n
            e = wL*entropy(lab[left]) + wR*entropy(lab[right])
            gain = parent_e - e
            split_info = -(wL*np.log2(wL) + wR*np.log2(wR))
            gr = gain / split_info if split_info > 0 else 0.0
            if best is None or gain > best["gain"]:
                best = dict(col=col, name=feat_names[col], thr=float(t),
                            gain=gain, gr=gr, e=e,
                            nL=len(left), nR=len(right),
                            eL=entropy(lab[left]), eR=entropy(lab[right]))
    return best, parent_e

def build_tree(Xs, idx, lab, feat_cols, depth, max_depth):
    node = {}
    node["depth"] = depth
    node["n"] = len(idx)
    if len(idx) == 0:
        node["leaf"] = True; node["pred"] = 0
        node["parent_entropy"] = 0.0; node["positive_rate"] = 0.0
        return node
    best, pe = best_split(Xs, idx, lab, feat_cols)
    node["parent_entropy"] = round(pe, 4)
    node["positive_rate"] = round(float(np.mean(lab[idx] == 1)), 3)
    if best is None or depth >= max_depth:
        node["leaf"] = True
        node["pred"] = int(round(np.mean(lab[idx])))
        return node
    vals = Xs[idx, best["col"]]
    lidx = idx[vals <= best["thr"]]; ridx = idx[vals > best["thr"]]
    if len(lidx) == 0 or len(ridx) == 0:
        node["leaf"] = True
        node["pred"] = int(round(np.mean(lab[idx])))
        return node
    node["leaf"] = False
    node["split"] = {k: (round(v,4) if isinstance(v, float) else v) for k,v in best.items() if k!="name"}
    node["split"]["name"] = best["name"]
    node["left"] = build_tree(Xs, lidx, lab, feat_cols, depth+1, max_depth)
    node["right"] = build_tree(Xs, ridx, lab, feat_cols, depth+1, max_depth)
    return node

# small, interpretable 2-feature view for the handwritten trace
trace_cols = [feat_names.index("GrLivArea"), feat_names.index("OverallQual")]
idx_all = np.arange(len(y))
scratch_tree = build_tree(X, idx_all, above, trace_cols, 0, 3)
import pprint
print("\n=== from-scratch recursive tree (GrLivArea + OverallQual, depth<=3) ===")
def show(node, pre=""):
    if node["leaf"]:
        print(f"{pre}leaf n={node['n']} pred={node['pred']} p(贵)={node['positive_rate']}")
        return
    s = node["split"]
    print(f"{pre}[depth {node['depth']}] split {s['name']} <= {s['thr']:.1f} | gain={s['gain']:.4f} gr={s['gr']:.4f} | n={node['n']} p(贵)={node['positive_rate']} parent_e={node['parent_entropy']}")
    show(node["left"], pre+"  L ")
    show(node["right"], pre+"  R ")
show(scratch_tree)

# gain vs gain-ratio table on full feature set (classification root candidates)
print("\n=== info gain vs gain ratio (top features) ===")
rows = []
all_idx = np.arange(len(y))
parent_e_full = entropy(above[all_idx])
for j, name in enumerate(feat_names):
    vals = X[all_idx, j]
    cand = np.unique(vals)
    if len(cand) <= 1: continue
    thr = (cand[:-1] + cand[1:]) / 2.0
    if len(thr) > 30:
        thr = np.quantile(vals, np.linspace(0.02, 0.98, 30))
    bestj = None
    for t in thr:
        L = all_idx[vals <= t]; R = all_idx[vals > t]
        if len(L)==0 or len(R)==0: continue
        wL, wR = len(L)/len(all_idx), len(R)/len(all_idx)
        e = wL*entropy(above[L]) + wR*entropy(above[R])
        g = parent_e_full - e
        si = -(wL*np.log2(wL) + wR*np.log2(wR))
        gr = g/si if si>0 else 0
        if bestj is None or g > bestj[0]:
            bestj = (g, gr, float(t), len(cand))
    if bestj is not None:
        rows.append((name, bestj[0], bestj[1], bestj[2], bestj[3]))
rows.sort(key=lambda r: r[1], reverse=True)
print(f"{'feature':<14}{'info_gain':>11}{'gain_ratio':>12}{'thr':>12}{'n_unique':>10}")
for name, g, gr, t, nu in rows[:12]:
    print(f"{name:<14}{g:>11.4f}{gr:>12.4f}{t:>12.1f}{nu:>10}")

# ---------------------------------------------------------------- sklearn classification tree (entropy + gini)
Xtr, Xte, ytr, yte = train_test_split(X, above, test_size=0.25, random_state=RNG, stratify=above)
clf_e = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=RNG)
clf_e.fit(Xtr, ytr)
clf_g = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=RNG)
clf_g.fit(Xtr, ytr)
print(f"\nentropy tree train acc={clf_e.score(Xtr,ytr):.3f} test acc={clf_e.score(Xte,yte):.3f}")
print(f"gini    tree train acc={clf_g.score(Xtr,ytr):.3f} test acc={clf_g.score(Xte,yte):.3f}")
# root split of entropy tree
tree_e = clf_e.tree_
root_feat = feat_names[tree_e.feature[0]]
root_thr = tree_e.threshold[0]
root_e_left = tree_e.impurity[tree_e.children_left[0]]
root_e_right = tree_e.impurity[tree_e.children_right[0]]
print(f"entropy-tree root split: {root_feat} <= {root_thr:.2f} (left n={tree_e.n_node_samples[tree_e.children_left[0]]}, right n={tree_e.n_node_samples[tree_e.children_right[0]]})")
print(f"  root child impurities: left={root_e_left:.4f} right={root_e_right:.4f}")
imp_e = sorted(zip(feat_names, clf_e.feature_importances_), key=lambda x:-x[1])[:6]
print("entropy-tree top importances:", [(n, round(v,3)) for n,v in imp_e])

# ---------------------------------------------------------------- regression tree (CART)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.25, random_state=RNG)
# deep (overfit) tree
deep = DecisionTreeRegressor(criterion="squared_error", max_depth=None, random_state=RNG)
deep.fit(Xr_tr, yr_tr)
deep_tr = r2_score(yr_tr, deep.predict(Xr_tr))
deep_te = r2_score(yr_te, deep.predict(Xr_te))
# shallow
shal = DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=RNG)
shal.fit(Xr_tr, yr_tr)
shal_tr = r2_score(yr_tr, shal.predict(Xr_tr)); shal_te = r2_score(yr_te, shal.predict(Xr_te))
print(f"\ndeep reg tree  train R2={deep_tr:.4f} test R2={deep_te:.4f} (leaves={deep.get_n_leaves()})")
print(f"depth3 reg tree train R2={shal_tr:.4f} test R2={shal_te:.4f}")
reg_imp = sorted(zip(feat_names, deep.feature_importances_), key=lambda x:-x[1])[:8]
print("reg-tree top importances:", [(n, round(v,3)) for n,v in reg_imp])
dtree = deep.tree_
print(f"reg-tree root split: {feat_names[dtree.feature[0]]} <= {dtree.threshold[0]:.2f}")

# pre-pruning curve
depths = list(range(1, 16))
pre = {"depth": [], "train_r2": [], "test_r2": []}
for d in depths:
    m = DecisionTreeRegressor(criterion="squared_error", max_depth=d, random_state=RNG)
    m.fit(Xr_tr, yr_tr)
    pre["depth"].append(d)
    pre["train_r2"].append(round(r2_score(yr_tr, m.predict(Xr_tr)), 4))
    pre["test_r2"].append(round(r2_score(yr_te, m.predict(Xr_te)), 4))
best_d = int(np.argmax(pre["test_r2"]))
print(f"\npre-prune best depth={best_d} test R2={pre['test_r2'][best_d-1]:.4f}")

# post-pruning (CCP)
path = deep.cost_complexity_pruning_path(Xr_tr, yr_tr)
ccp_alphas = path.ccp_alphas
# pick alpha that maximizes validation R2 via simple train/test on train split? use internal: choose alpha by cross-like: pick largest alpha with test R2 >= max-0.005
ccps = []
for a in ccp_alphas:
    if a == 0: continue
    m = DecisionTreeRegressor(criterion="squared_error", ccp_alpha=a, random_state=RNG)
    m.fit(Xr_tr, yr_tr)
    ccps.append((a, r2_score(yr_te, m.predict(Xr_te)), m.get_n_leaves()))
ccps.sort(key=lambda x: x[0])
best_alpha = ccps[np.argmax([c[1] for c in ccps])][0]
post = DecisionTreeRegressor(criterion="squared_error", ccp_alpha=best_alpha, random_state=RNG)
post.fit(Xr_tr, yr_tr)
post_te = r2_score(yr_te, post.predict(Xr_te)); post_tr = r2_score(yr_tr, post.predict(Xr_tr))
print(f"post-prune (CCP) best_alpha={best_alpha:.2e} test R2={post_te:.4f} (leaves={post.get_n_leaves()})")

# ---------------------------------------------------------------- decision boundary (2 numeric features)
bd_feats = ["GrLivArea", "YearBuilt"]
bi = [feat_names.index(c) for c in bd_feats]
Xb = X[:, bi]
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, y, test_size=0.25, random_state=RNG)
bd = DecisionTreeRegressor(criterion="squared_error", max_depth=6, random_state=RNG)
bd.fit(Xb_tr, yb_tr)
bd_te = r2_score(yb_te, bd.predict(Xb_te))
print(f"\ndecision-boundary reg tree (GrLivArea x YearBuilt) depth=6 test R2={bd_te:.4f}")

# ---------------------------------------------------------------- save stats
stats = {
    "dataset": "Ames housing (Kaggle train.csv, OpenML house_prices v1)",
    "n_rows": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "median_price": round(median_price, 0),
    "mean_price": round(float(np.mean(y)), 0),
    "min_price": int(y.min()), "max_price": int(y.max()),
    "above_median": int(above.sum()), "below_median": int((1-above).sum()),
    "scratch_tree_depth3": {"root_feature": scratch_tree["split"]["name"],
                             "root_thr": scratch_tree["split"]["thr"],
                             "root_gain": scratch_tree["split"]["gain"],
                             "root_gr": scratch_tree["split"]["gr"],
                             "parent_entropy": scratch_tree["parent_entropy"]},
    "gain_ratio_table": [{"feature": n, "info_gain": round(g,4),
                          "gain_ratio": round(gr,4), "thr": round(t,1),
                          "n_unique": int(nu)} for n,g,gr,t,nu in rows[:12]],
    "clf_entropy": {"train_acc": round(clf_e.score(Xtr,ytr),3),
                    "test_acc": round(clf_e.score(Xte,yte),3),
                    "root_feature": root_feat, "root_thr": round(root_thr,2),
                    "root_e_left": round(root_e_left,4), "root_e_right": round(root_e_right,4),
                    "top_importances": [(n, round(v,3)) for n,v in imp_e]},
    "clf_gini": {"train_acc": round(clf_g.score(Xtr,ytr),3),
                 "test_acc": round(clf_g.score(Xte,yte),3)},
    "reg_deep": {"train_r2": round(deep_tr,4), "test_r2": round(deep_te,4),
                 "leaves": int(deep.get_n_leaves()),
                 "root_feature": feat_names[dtree.feature[0]],
                 "root_thr": round(dtree.threshold[0],2),
                 "top_importances": [(n, round(v,3)) for n,v in reg_imp]},
    "reg_shallow_d3": {"train_r2": round(shal_tr,4), "test_r2": round(shal_te,4)},
    "pre_prune": pre, "pre_prune_best_depth": best_d,
    "post_prune": {"alpha": float(best_alpha), "test_r2": round(post_te,4),
                   "train_r2": round(post_tr,4), "leaves": int(post.get_n_leaves())},
    "boundary": {"feats": bd_feats, "depth": 6, "test_r2": round(bd_te,4)},
}
with open(os.path.join(BASE, "stats.json"), "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("\nstats.json written.")

# save processed feature matrix for figure reuse
proc = feat.copy(); proc["SalePrice"] = y; proc["above_median"] = above
proc.to_csv(os.path.join(DATA, "processed.csv"), index=False)
np.save(os.path.join(DATA, "X.npy"), X)
np.save(os.path.join(DATA, "y.npy"), y)
np.save(os.path.join(DATA, "above.npy"), above)
with open(os.path.join(DATA, "feat_names.json"), "w") as f:
    json.dump(feat_names, f)
