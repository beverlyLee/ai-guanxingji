"""
M9 实验主体（无监督聚类：KMeans + DBSCAN）

1) Gap 统计量（三种参照系）：这批规格数据到底该不该聚、该聚几档
2) KMeans：肘部法 + 轮廓系数选 K，看两个指标会不会打架
3) DBSCAN：多空间扫描 + 超几何检验，看噪声里有没有富集异类机型
4) 鲁棒性对照：换预处理方式，结论会不会翻

输入：data/phone_clean.csv（3104 款手机，2015–2022，由 prepare_data.py 生成）
输出：stats.json（全部实测数值）、data/phone_clustered.csv、data/*.npy

运行：python code/experiment.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, skew
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RNG = 42

stats = {}


def log(*a):
    print(*a, flush=True)


# ============ 1. 载入与特征变换 ============
d = pd.read_csv(ROOT / "data" / "phone_clean.csv")
log(f"样本：{len(d)} 款手机，{int(d['year'].min())}–{int(d['year'].max())}")

NUM = ["price_eur", "screen_in", "weight_g", "battery_mah", "ppi",
       "storage_gb", "ram_gb", "camera_mp"]
CN = {"price_eur": "价格(欧元)", "screen_in": "屏幕(英寸)", "weight_g": "重量(克)",
      "battery_mah": "电池(mAh)", "ppi": "像素密度(ppi)", "storage_gb": "存储(GB)",
      "ram_gb": "内存(GB)", "camera_mp": "主摄(MP)"}

sk = d[NUM].apply(lambda s: skew(s.dropna())).sort_values(ascending=False)
log("\n===== 各特征偏度（>1 视为右偏严重，取对数）=====")
log(sk.round(2).to_string())

LOGGED = [c for c in NUM if sk[c] > 1.0]
stats["logged_features"] = LOGGED
stats["skew_raw"] = {k: round(float(v), 3) for k, v in sk.items()}
log(f"\n取对数的特征：{LOGGED}")

X = d[NUM].copy()
for c in LOGGED:
    X[c] = np.log1p(X[c])
stats["skew_after_log"] = {k: round(float(v), 3)
                           for k, v in X.apply(lambda s: skew(s.dropna())).items()}

Z = StandardScaler().fit_transform(X.values)   # 不做这步，聚类就变成"谁的数值大谁说话"
log(f"标准化后：均值最大绝对值 {np.abs(Z.mean(axis=0)).max():.2e}，标准差 {Z.std(axis=0).mean():.4f}")

# ============ 2. Gap 统计量：到底该不该聚、该聚几档 ============
# 思路：把"真实数据的簇内紧凑度"和"同样大小的均匀随机数据"比。若真实数据并不比随机
# 数据更紧凑，就说明没有超出随机的结构，K=1（不聚）就够了。B 次参照取均值与标准差。
def log_within_k(Xm, labels_):
    """簇内成对距离和 / (2·n_k)，再对各簇求和（Tibshirani 的 log W_k）。"""
    tot = 0.0
    for c in np.unique(labels_):
        idx = np.where(labels_ == c)[0]
        if len(idx) < 2:
            continue
        D = pairwise_distances(Xm[idx])
        tot += D.sum() / (2 * len(idx))
    return tot


def make_reference(Xm, method, rng):
    """生成参照数据。均匀盒参照会把"真实数据本来就更紧凑"误算成结构，故多试两种。"""
    n = Xm.shape[0]
    if method == "uniform_box":          # 逐维均匀（最弱的原假设）
        return rng.uniform(Xm.min(axis=0), Xm.max(axis=0), size=Xm.shape)
    if method == "uniform_pca":          # 沿主成分轴均匀，保住相关结构
        p = PCA().fit(Xm)
        S = p.transform(Xm)
        Sr = rng.uniform(S.min(axis=0), S.max(axis=0), size=S.shape)
        return p.inverse_transform(Sr)
    if method == "gaussian":             # 多元高斯，保住均值与协方差
        return rng.multivariate_normal(Xm.mean(axis=0),
                                       np.cov(Xm, rowvar=False) + 1e-9 * np.eye(Xm.shape[1]),
                                       size=n)
    raise ValueError(method)


def gap_statistic(Xm, k_list, B=20, seed=RNG, method="uniform_pca"):
    rng = np.random.RandomState(seed)
    out = {}
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xm)
        logwk = np.log(log_within_k(Xm, km.labels_))
        refs = []
        for b in range(B):
            Xr = make_reference(Xm, method, rng)
            kmr = KMeans(n_clusters=k, n_init=5, random_state=seed + b).fit(Xr)
            refs.append(np.log(log_within_k(Xr, kmr.labels_)))
        refs = np.array(refs)
        out[k] = {"logW": float(logwk), "ref_mean": float(refs.mean()),
                  "gap": float(refs.mean() - logwk), "sd": float(refs.std(ddof=1))}
    return out


def pick_k(g, k_list, B=20):
    """Tibshirani 规则：最小的 K 使 gap(K) ≥ gap(K+1) − s(K+1)。取不到就说明范围内没有拐点。"""
    for k in k_list[:-1]:
        s_next = g[k + 1]["sd"] * np.sqrt(1 + 1 / B)
        if g[k]["gap"] >= g[k + 1]["gap"] - s_next:
            return k
    return None


K_GRID_GAP = list(range(1, 9))
stats["gap_statistic"] = {}
for method in ("uniform_box", "uniform_pca", "gaussian"):
    g = gap_statistic(Z, K_GRID_GAP, B=20, method=method)
    log(f"\n===== Gap 统计量 · 参照 = {method}（B=20）=====")
    log(f"{'K':>3} {'logW':>9} {'参照均值':>10} {'gap':>8} {'sd':>8}")
    for k, v in g.items():
        log(f"{k:3d} {v['logW']:9.1f} {v['ref_mean']:10.1f} {v['gap']:8.3f} {v['sd']:8.4f}")
    kk = pick_k(g, K_GRID_GAP)
    log(f"→ 该参照下选出的 K = {kk if kk else '范围内无拐点（gap 一直单调上升）'}")
    stats["gap_statistic"][method] = {
        "table": {str(k): {a: round(b, 4) for a, b in v.items()} for k, v in g.items()},
        "k_selected": kk,
    }

# ============ 3. PCA ============
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n90 = int(np.searchsorted(cum, 0.90) + 1)
PC = pca.transform(Z)
log("\n===== PCA 解释方差 =====")
for i, (e, c) in enumerate(zip(evr, cum), 1):
    log(f"PC{i}: {e*100:5.1f}%  累计 {c*100:5.1f}%")
log(f"累计达 90% 需要 {n90} 个主成分；前 2 个合计 {cum[1]*100:.1f}%")
stats["pca_explained_variance_ratio"] = [round(float(x), 4) for x in evr]
stats["pca_n_components_for_90pct"] = n90
stats["pca_loadings_pc1"] = {CN[c]: round(float(v), 3)
                             for c, v in zip(NUM, pca.components_[0])}
stats["pca_loadings_pc2"] = {CN[c]: round(float(v), 3)
                             for c, v in zip(NUM, pca.components_[1])}
log("\nPC1 载荷（谁在主导这条轴）：")
for c, v in sorted(stats["pca_loadings_pc1"].items(), key=lambda kv: -abs(kv[1])):
    log(f"  {c:14s} {v:+.3f}")

# DBSCAN 输入用前 3 主成分：8 维空间里密度会被维度稀释，先降到 3 维再谈"密不密"
Z_db = PC[:, :3]

# ============ 4. KMeans ============
log("\n===== KMeans 选 K =====")
ks = list(range(2, 11))
inertia, sil = [], []
km_models = {}
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Z)
    inertia.append(float(km.inertia_))
    s = float(silhouette_score(Z, km.labels_))
    sil.append(s)
    km_models[k] = km
    log(f"K={k:2d}  惯性={km.inertia_:9.1f}  轮廓系数={s:.4f}")

inertia, sil = np.array(inertia), np.array(sil)
drop_rate = (np.diff(inertia) * -1)
drop_rate = drop_rate / drop_rate[0]
k_elbow = int(ks[int(np.argmax(drop_rate[:-1] - drop_rate[1:])) + 1])
k_sil = int(ks[int(np.argmax(sil))])
log(f"\n肘部法给出 K = {k_elbow}（惯性下降率从这一档起明显放缓）")
log(f"轮廓系数最大在 K = {k_sil}（{sil.max():.4f}）")

stats["kmeans"] = {
    "k_grid": ks,
    "inertia": [round(float(x), 1) for x in inertia],
    "silhouette": [round(float(x), 4) for x in sil],
    "k_by_elbow": k_elbow,
    "k_by_silhouette": k_sil,
    "silhouette_max": round(float(sil.max()), 4),
    "indicators_agree": bool(k_elbow == k_sil),
}

# 主用 K=4（肘部法），因为市场自己的语言就是"入门/中端/高端/旗舰"四档；
# 同时把 K=2、K=3 的画像一并给出，展示"档位是人切的"
K_MAIN = k_elbow
def profile(labels_, tag):
    g = d.groupby(labels_)
    p = g[NUM].mean()
    p["款数"] = g.size()
    p["欧莱屏占比"] = d.assign(v=d["is_oled"]).groupby(labels_)["v"].mean()
    p["5G占比"] = d.assign(v=d["is_5g"]).groupby(labels_)["v"].mean()
    p["价格中位"] = g["price_eur"].median()
    p = p.sort_values("price_eur")
    log(f"\n----- {tag} 各簇画像（按均价升序）-----")
    log(p.round(2).to_string())
    return p


profiles = {}
for k in (2, 3, 4):
    lab = km_models[k].labels_
    p = profile(lab, f"K={k}")
    profiles[k] = p
    stats[f"profile_k{k}"] = {
        "sizes": {str(int(i)): int(p.loc[i, "款数"]) for i in p.index},
        "price_means": [round(float(x), 1) for x in p["price_eur"]],
        "price_medians": [round(float(x), 0) for x in p["价格中位"]],
        "battery": [round(float(x), 0) for x in p["battery_mah"]],
        "ram": [round(float(x), 2) for x in p["ram_gb"]],
        "storage": [round(float(x), 0) for x in p["storage_gb"]],
        "screen": [round(float(x), 2) for x in p["screen_in"]],
        "weight": [round(float(x), 0) for x in p["weight_g"]],
        "camera": [round(float(x), 1) for x in p["camera_mp"]],
        "ppi": [round(float(x), 0) for x in p["ppi"]],
        "oled_share": [round(float(x), 2) for x in p["欧莱屏占比"]],
        "g5_share": [round(float(x), 2) for x in p["5G占比"]],
    }

labels = km_models[K_MAIN].labels_
stats["kmeans"]["k_main"] = K_MAIN

# 各簇价格区间是否真的分开：分位数重叠说明"档"之间没有鸿沟
g = d.assign(cl=labels).groupby("cl")["price_eur"]
q = g.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
q.columns = ["p05", "p25", "p50", "p75", "p95"]
q = q.loc[profiles[K_MAIN].index]
log(f"\nK={K_MAIN} 各簇价格分位（看相邻簇区间有没有重叠）：")
log(q.round(0).to_string())
stats["price_quantiles_k_main"] = {
    str(int(i)): {c: float(q.loc[i, c]) for c in q.columns} for i in q.index
}
overlap = []
order = list(profiles[K_MAIN].index)
for a, b in zip(order[:-1], order[1:]):
    lo_a, hi_a = q.loc[a, "p05"], q.loc[a, "p95"]
    lo_b, hi_b = q.loc[b, "p05"], q.loc[b, "p95"]
    ov = min(hi_a, hi_b) - max(lo_a, lo_b)
    overlap.append(round(float(ov), 1))
stats["price_band_overlap_width"] = overlap
log(f"相邻簇价格区间重叠宽度：{overlap}")

# ============ 5. DBSCAN：多空间扫描 + 噪声富集检验 ============
log("\n===== DBSCAN =====")
FOLD = "Fold|Flip|Razr|Mate X|Mix Fold|V Fold|Surface Duo"
RUGGED = "Doogee|Ulefone|Blackview|Oukitel|AGM|Power Armor"
d["is_fold"] = d["model"].str.contains(FOLD, case=False, na=False).astype(int)
d["is_rugged"] = d["model"].str.contains(RUGGED, case=False, na=False).astype(int)
d["is_odd"] = ((d["is_fold"] == 1) | (d["is_rugged"] == 1)).astype(int)
N = len(d)
K_odd = int(d["is_odd"].sum())
log(f"样本内折叠屏 {int(d['is_fold'].sum())} 款，三防/巨电池机 {int(d['is_rugged'].sum())} 款，"
    f"并集 {K_odd} 款（占 {K_odd/N*100:.1f}%）")


def knee_eps(space, min_samples):
    """k-距离曲线拐点 → eps（kneedle：归一化后取离首尾连线最远的点）。"""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(space)
    kd = np.sort(nn.kneighbors(space)[0][:, -1])
    xs = np.arange(len(kd))
    xn = (xs - xs.min()) / (xs.max() - xs.min())
    yn = (kd - kd.min()) / (kd.max() - kd.min())
    line = yn[0] + (yn[-1] - yn[0]) * xn
    i = int(np.argmax(line - yn))
    return kd, i, round(float(kd[i]), 3)


def run_db(space, eps, ms):
    lab = DBSCAN(eps=eps, min_samples=ms).fit(space).labels_
    n_clu = len(set(lab)) - (1 if -1 in lab else 0)
    noise = lab == -1
    n_noise = int(noise.sum())
    s = np.nan
    if n_clu >= 2 and n_noise < len(lab):
        m = ~noise
        if len(set(lab[m])) >= 2:
            s = float(silhouette_score(space[m], lab[m]))
    return lab, n_clu, n_noise, s, noise


def enrich(noise_mask):
    """噪声里"异类机型"是否被富集：观测/期望 与超几何检验的 p 值。"""
    obs = int((noise_mask & (d["is_odd"] == 1)).sum())
    n_noise = int(noise_mask.sum())
    exp = n_noise * K_odd / N
    p = float(hypergeom.sf(obs - 1, N, K_odd, n_noise)) if n_noise and obs else float("nan")
    return {"observed": obs, "expected": round(exp, 1),
            "enrichment": round(obs / exp, 2) if exp else None,
            "p_value": p, "noise_n": n_noise}


# 候选空间：8 维原始标准化 + 主成分子空间（含白化，让各主成分等权）
eig = pca.explained_variance_
SPACES = {
    "8维标准化": Z,
    "前3主成分": PC[:, :3],
    "前3主成分(白化)": PC[:, :3] / np.sqrt(eig[:3]),
    "前2主成分(白化)": PC[:, :2] / np.sqrt(eig[:2]),
}

rows, results = [], {}
for name, space in SPACES.items():
    ms = max(2 * space.shape[1], 4)          # 经验值：2 × 维度
    kd, ki, eps = knee_eps(space, ms)
    lab, n_clu, n_noise, s, noise = run_db(space, eps, ms)
    e = enrich(noise)
    results[name] = dict(space=space, eps=eps, ms=ms, lab=lab, n_clu=n_clu,
                         n_noise=n_noise, sil=s, enrich=e, knee_pct=ki / len(kd) * 100)
    rows.append(dict(空间=name, 维度=space.shape[1], min_samples=ms, eps_knee=eps,
                     拐点分位=f"{ki/len(kd)*100:.0f}%", 簇数=n_clu,
                     噪声数=n_noise, 噪声占比=f"{n_noise/N*100:.1f}%",
                     轮廓系数=None if np.isnan(s) else round(s, 4),
                     异类富集倍数=e["enrichment"], p值=f"{e['p_value']:.1e}"))
    log(f"\n[{name}] 维度={space.shape[1]} min_samples={ms} eps={eps}"
        f"（拐点 {ki/len(kd)*100:.0f}%） → {n_clu} 簇，噪声 {n_noise} 款"
        f"（{n_noise/N*100:.1f}%），轮廓系数 {s:.3f}，异类富集 {e['enrichment']}×"
        f"，超几何 p={e['p_value']:.1e}")

log("\n===== DBSCAN 各空间汇总 =====")
log(pd.DataFrame(rows).to_string(index=False))
stats["dbscan_space_sweep"] = rows

# 选主设定：先把判据表在前面（轮廓系数 > 0 且噪声率 3%–15%），再取轮廓系数最高者
ok = [n for n, r in results.items()
      if (not np.isnan(r["sil"])) and r["sil"] > 0 and 0.03 <= r["n_noise"] / N <= 0.15]
log(f"\n满足'轮廓系数>0 且噪声率 3%–15%'的空间：{ok if ok else '无'}")
MAIN = max(ok, key=lambda n: results[n]["sil"]) if ok else None
if MAIN is None:      # 没有合格设定时，退而用噪声富集倍数最高者，并如实标注
    MAIN = max(results, key=lambda n: results[n]["enrich"]["enrichment"] or 0)
    log(f"无合格设定，退取噪声富集最高者：{MAIN}")
log(f"主设定空间 = {MAIN}")

R = results[MAIN]
min_samples = R["ms"]
EPS = R["eps"]
Z_db = R["space"]
d["db_label"] = R["lab"]
n_clu, n_noise, s_db = R["n_clu"], R["n_noise"], R["sil"]

# 主设定的 eps 网格（看结论对 eps 有多敏感）
grid = []
for eps in np.arange(0.3, 3.01, 0.1):
    lab_g, nc, nn_, s, _ = run_db(Z_db, float(eps), min_samples)
    grid.append(dict(eps=round(float(eps), 2), n_clusters=nc, n_noise=nn_,
                     noise_pct=round(nn_ / N * 100, 2),
                     silhouette=None if np.isnan(s) else round(s, 4)))
log(f"\n主设定（{MAIN}）的 eps 网格：")
log(pd.DataFrame(grid).to_string(index=False))
stats["dbscan_eps_grid_main_space"] = {"space": MAIN, "grid": grid}

stats["dbscan"] = {"space": MAIN, "min_samples": min_samples, "eps": EPS,
                   "n_clusters": n_clu, "n_noise": n_noise,
                   "noise_pct": round(n_noise / N * 100, 2),
                   "silhouette_non_noise": None if np.isnan(s_db) else round(s_db, 4),
                   "enrichment": R["enrich"], "knee_percentile": round(R["knee_pct"], 1)}
log(f"\n主设定：eps={EPS} → {n_clu} 簇，噪声 {n_noise} 款（{n_noise/N*100:.1f}%），"
    f"轮廓系数 {s_db:.3f}")

# 8 维对照：同样的 eps 与 min_samples
_, n_clu8, n_noise8, s8, _ = run_db(Z, EPS, min_samples)
stats["dbscan_8d_control"] = {"eps": EPS, "min_samples": min_samples,
                              "n_clusters": n_clu8, "n_noise": n_noise8,
                              "noise_pct": round(n_noise8 / N * 100, 2),
                              "silhouette_non_noise": None if np.isnan(s8) else round(s8, 4)}
log(f"对照：同参数直接在 8 维标准化空间跑 → {n_clu8} 簇，噪声 {n_noise8} 款"
    f"（{n_noise8/N*100:.1f}%），轮廓系数 {s8:.3f}")

noise_mask = d["db_label"] == -1
cross = {"fold_total": int(d["is_fold"].sum()),
         "fold_in_noise": int((noise_mask & (d["is_fold"] == 1)).sum()),
         "rugged_total": int(d["is_rugged"].sum()),
         "rugged_in_noise": int((noise_mask & (d["is_rugged"] == 1)).sum()),
         "odd_total": K_odd, "odd_in_noise": int((noise_mask & (d["is_odd"] == 1)).sum())}
log("\n===== 噪声点构成 =====")
log(f"折叠屏     {cross['fold_in_noise']:3d}/{cross['fold_total']:3d} 落入噪声"
    f"（{cross['fold_in_noise']/max(cross['fold_total'],1)*100:.0f}%）")
log(f"三防巨电池  {cross['rugged_in_noise']:3d}/{cross['rugged_total']:3d} 落入噪声"
    f"（{cross['rugged_in_noise']/max(cross['rugged_total'],1)*100:.0f}%）")
log(f"两类并集    {cross['odd_in_noise']:3d}/{cross['odd_total']:3d} 落入噪声"
    f"，占全部噪声的 {cross['odd_in_noise']/max(n_noise,1)*100:.1f}%")
log(f"随机抽同样多的机型，期望只会命中 {K_odd/N*n_noise:.1f} 款 → "
    f"富集 {R['enrich']['enrichment']}×，超几何检验 p = {R['enrich']['p_value']:.2e}")

top_noise = d[noise_mask].nlargest(12, "price_eur")[
    ["model", "price_eur", "screen_in", "weight_g", "battery_mah", "is_fold"]]
log("\n噪声里最贵的 12 款：")
log(top_noise.to_string(index=False))
stats["noise_composition"] = cross
stats["noise_top_price"] = top_noise.to_dict("records")

# 对照：KMeans 把这些异类塞进了哪些簇（硬聚类的"平均化"）
fold_km = pd.Series(labels[d["is_fold"].values == 1]).value_counts().to_dict()
fold_db = d.loc[d["is_fold"] == 1, "db_label"].value_counts().to_dict()
log(f"\n折叠屏在 KMeans(K={K_MAIN}) 里的分布：{fold_km}")
log(f"折叠屏在 DBSCAN 里的分布：{fold_db}")
stats["fold_km_distribution"] = {str(k): int(v) for k, v in fold_km.items()}
stats["fold_db_distribution"] = {str(k): int(v) for k, v in fold_db.items()}

# ============ 6. 鲁棒性对照：换预处理，结论会不会翻 ============
log("\n===== 鲁棒性对照：换预处理方式，'结构弱'这个结论会不会翻 =====")
rob = {}
variants = {
    "log+标准化（本文采用）": Z,
    "仅标准化（不取对数）": StandardScaler().fit_transform(d[NUM].values),
    "只用 4 个特征（价格/屏幕/电池/内存）": StandardScaler().fit_transform(
        d[["price_eur", "screen_in", "battery_mah", "ram_gb"]].apply(np.log1p).values),
    "只用 4 个特征且不取对数": StandardScaler().fit_transform(
        d[["price_eur", "screen_in", "battery_mah", "ram_gb"]].values),
}
for name, Zm in variants.items():
    row = {}
    for k in (2, 3, 4, 5):
        kmv = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Zm)
        row[k] = round(float(silhouette_score(Zm, kmv.labels_)), 4)
    g2 = gap_statistic(Zm, [1, 2], B=10)
    rob[name] = {"silhouette": row, "gap_k2_minus_k1": round(g2[2]["gap"] - g2[1]["gap"], 3)}
    log(f"{name:32s} 轮廓系数 K=2..5 {list(row.values())}  "
        f"gap(2)-gap(1)={g2[2]['gap'] - g2[1]['gap']:+.3f}")
stats["robustness"] = rob

np.save(ROOT / "data" / "Z.npy", Z)
np.save(ROOT / "data" / "PC.npy", PC)
np.save(ROOT / "data" / "labels_km.npy", labels)
np.save(ROOT / "data" / "labels_db.npy", d["db_label"].values)
d.to_csv(ROOT / "data" / "phone_clustered.csv", index=False)

with open(ROOT / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
log(f"\n已写出 {ROOT/'stats.json'} 与 data/phone_clustered.csv")
