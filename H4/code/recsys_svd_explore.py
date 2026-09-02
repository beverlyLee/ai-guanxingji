#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI观星记 · 线性代数篇（推荐系统 / 信息茧房）SVD 探索脚本
数据集：MovieLens 100K（GroupLens，公开可下载，零编造）
目的：用真实数据跑通 矩阵 -> 中心化 -> SVD -> 奇异值/方差解释率 -> 潜空间 biplot，
      并用"留出验证的评分预测误差"与"推荐类型多样性熵"两条真实曲线，
      为 H4 正文与配图提供可复现的真实数字。
所有图从原始数据文件重生，无外部缓存。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 字体（中文用 STHeiti，缺失则回退） ----------
CN_FONT = "/System/Library/Fonts/STHeiti Light.ttc"
if os.path.exists(CN_FONT):
    font_manager.fontManager.addfont(CN_FONT)
    cn_name = font_manager.FontProperties(fname=CN_FONT).get_name()
    plt.rcParams["font.family"] = cn_name
    print(f"已注册中文字体：{cn_name}")
else:
    print("未找到 STHeiti，回退 DejaVu Sans（中文可能缺字）")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

DATA = "data/ml-100k"
OUT = "assets"

# ---------- 1. 读数据 ----------
ratings = pd.read_csv(f"{DATA}/u.data", sep="\t",
                      names=["user", "item", "rating", "ts"])
items = pd.read_csv(f"{DATA}/u.item", sep="|", encoding="latin-1",
                    usecols=range(24),
                    names=["movie_id", "title", "release", "video", "url"] +
                          ["g%d" % i for i in range(19)])
GENRES = ["Action","Adventure","Animation","Children's","Comedy","Crime",
          "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
          "Mystery","Romance","Sci-Fi","Thriller","War","Western"]
GENRE_COLS = ["g%d" % i for i in range(19)]
genre_mat = items[GENRE_COLS].values.astype(float)   # n_items × 18

n_users = ratings["user"].nunique()
n_items = ratings["item"].nunique()
print(f"原始规模：{n_users} 用户 × {n_items} 电影，{len(ratings)} 条评分")

u_idx = {u: i for i, u in enumerate(sorted(ratings["user"].unique()))}
i_idx = {m: j for j, m in enumerate(sorted(ratings["item"].unique()))}
ratings["ui"] = ratings["user"].map(u_idx)
ratings["mi"] = ratings["item"].map(i_idx)

# ---------- 2. 建评分矩阵并中心化（列均值） ----------
R = np.full((n_users, n_items), np.nan)
R[ratings["ui"].values, ratings["mi"].values] = ratings["rating"].values
col_mean = np.nanmean(R, axis=0)
Rc = R - col_mean

# ---------- 3. 留出验证：抽 10% 已评分作测试集，其余训练 ----------
rng = np.random.default_rng(42)
obs = ratings[["ui", "mi", "rating"]].values
n_test = max(1, int(0.1 * len(obs)))
test_idx = rng.choice(len(obs), n_test, replace=False)
test_set = obs[test_idx]
train_mask = np.ones(len(obs), dtype=bool)
train_mask[test_idx] = False
train_set = obs[train_mask]

Rtr = np.full((n_users, n_items), 0.0)
Rtr[train_set[:, 0].astype(int), train_set[:, 1].astype(int)] = train_set[:, 2] - col_mean[train_set[:, 1].astype(int)]
sparsity = 1 - len(train_set) / (n_users * n_items)
print(f"训练稀疏度（缺失比例）：{sparsity:.3f}")

U, s, Vt = np.linalg.svd(Rtr, full_matrices=False)
var = s ** 2
cum = np.cumsum(var) / np.sum(var)
print("\n奇异值累计方差解释率：")
for k in [1, 2, 5, 10, 20, 50]:
    print(f"  前 {k:2d} 个隐因子 -> {cum[k-1]*100:.1f}%")

# 留出验证：用前 k 个因子重建，看测试集评分预测误差（RMSE）
print("\n留出验证 RMSE（训练集 SVD 重建未看电影评分，k 个隐因子）：")
rmse = {}
for k in [1, 2, 5, 10, 20, 50]:
    Rhat = (U[:, :k] * s[:k]) @ Vt[:k, :] + col_mean
    Rhat = np.clip(Rhat, 1, 5)
    err = Rhat[test_set[:, 0].astype(int), test_set[:, 1].astype(int)] - test_set[:, 2]
    e = np.sqrt(np.mean(err ** 2))
    rmse[k] = e
    print(f"  k={k:2d} -> RMSE = {e:.3f}（原始评分标准差≈1.1）")

# ---------- 4. fig1：评分矩阵热图（抽样块） ----------
fig, ax = plt.subplots(figsize=(6, 6))
block = Rc[:60, :60]
im = ax.imshow(block, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_title("评分矩阵的一角（60×60 用户×电影，蓝=高于均值/红=低于）")
ax.set_xlabel("电影（索引）"); ax.set_ylabel("用户（索引）")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="评分−列均值")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_rating_matrix.png"); plt.close(fig)

# ---------- 5. fig2：奇异值谱（左）+ 留出验证 RMSE（右） ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
axL.plot(range(1, 51), s[:50], "o-", color="#0072B2", ms=3, lw=1.2)
axL.set_xlabel("隐因子序号 k"); axL.set_ylabel("奇异值 σ")
axL.set_title("SVD 碎石图：能量集中在前几十个因子")
axR.plot(list(rmse.keys()), list(rmse.values()), "s-", color="#D55E00", ms=5, lw=1.5)
axR.axhline(1.1, ls="--", color="gray", lw=0.8)
axR.set_xlabel("隐因子数 k"); axR.set_ylabel("测试集评分预测 RMSE")
axR.set_title("用 k 个隐因子就能猜中你没看过的电影评分")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_scree_rmse.png"); plt.close(fig)

# ---------- 6. fig3：电影 biplot（潜空间类型抱团） ----------
# 注：信息茧房（推荐类型多样性熵）已单独拆到 cocoon_simulation.py -> fig4_cocoon_entropy.png，
#     本脚本只负责「潜空间 biplot」这一张，避免与 fig4 重复。
movies_pc = Vt[:2, :]
primary = items[GENRE_COLS].values.argmax(axis=1)
palette = ["#0072B2","#E69F00","#009E73","#CC79A7","#D55E00",
           "#56B4E9","#F0E442","#999999","#000000","#882255",
           "#AA4499","#332288","#44AA99","#117733","#DDCC77",
           "#661100","#6699CC","#88CCEE"]
fig, axL = plt.subplots(figsize=(7.5, 6))
for g in range(18):
    m = primary == g
    axL.scatter(movies_pc[0, m], movies_pc[1, m], s=6, alpha=0.35,
                color=palette[g], label=GENRES[g])
axL.set_xlabel("PC1（第 1 主口味轴）"); axL.set_ylabel("PC2（第 2 主口味轴）")
axL.set_title("电影被 SVD 压进 2 维潜空间，类型自然抱团")
famous = {"Toy Story (1995)": None, "Star Wars (1977)": None,
          "Terminator, The (1984)": None, "Pretty Woman (1990)": None,
          "Jurassic Park (1993)": None, "Lion King, The (1994)": None,
          "Silence of the Lambs, The (1991)": None, "Aladdin (1992)": None}
for t in famous:
    row = items.index[items["title"] == t]
    if len(row):
        j = row[0]
        axL.annotate(t.replace(" (", "\n("), (movies_pc[0, j], movies_pc[1, j]),
                     fontsize=6, xytext=(3, 3), textcoords="offset points")
axL.legend(fontsize=6, loc="upper right", framealpha=0.6, ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_biplot.png"); plt.close(fig)

print("\n三张主图已写入 assets/：fig1_rating_matrix / fig2_scree_rmse / fig3_biplot")
print("信息茧房配图 fig4_cocoon_entropy.png 由 cocoon_simulation.py 单独生成。")
