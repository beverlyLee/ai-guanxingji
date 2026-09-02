#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI观星记 · 线性代数篇（H4）· 信息茧房账 · 独立可复现脚本
==================================================================
对应正文章节「Q4 第三笔：茧房账」。
把"推荐系统如何把人越推越窄"这一个实验，单独拎出来写成一份
干净可跑、可出图的代码——不依赖其他图，只复现这一笔的数字。

实验设定（与正文逐字对齐）：
  - 数据集：MovieLens 100K（943 用户 × 1682 电影，10 万条评分）
  - 选一个中等活跃用户 user#615，已看过正好 43 部电影
  - 每轮用 SVD 口味尺子挑出 20 部"最懂他"的电影作推荐列表
  - 量这个列表的"类型多样性熵"（越单一，熵越低）
  - 让"用户看下最贴近的一部"→ 口味画像被往已知方向推 → 下一轮重算
  - 跑 25 轮，看熵怎么坍缩

可复现数字（用本机 data/ml-100k，random seed=42）：
  随机挑 20 部基线熵 = 3.12 bit（满多样性）
  用户 #615 初始（第 1 轮）熵 ≈ 2.65 bit
  25 轮后，最后 10 轮平均熵 = 2.47 bit
  相对随机基线收窄 = 20.7%
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
else:
    print("未找到 STHeiti，回退 DejaVu Sans（中文可能缺字）")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

DATA = "data/ml-100k"
OUT = "assets"
N_ROUNDS = 25          # 推荐轮数
TOPN = 20              # 每轮推荐列表长度
USER_ID = 616          # 中等活跃用户（矩阵 row index 615）：恰好看过 43 部

# ---------- 1. 读数据 ----------
ratings = pd.read_csv(f"{DATA}/u.data", sep="\t",
                      names=["user", "item", "rating", "ts"])
items = pd.read_csv(f"{DATA}/u.item", sep="|", encoding="latin-1",
                    usecols=range(24),
                    names=["movie_id", "title", "release", "video", "url"] +
                          ["g%d" % i for i in range(19)])
GENRE_COLS = ["g%d" % i for i in range(19)]
genre_mat = items[GENRE_COLS].values.astype(float)   # n_items × 19(哑变量, 含一列全0)

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

# ---------- 3. 留出验证：抽 10% 已评分作测试集，其余训练（与 recsys_svd_explore.py 同 seed） ----------
rng = np.random.default_rng(42)
obs = ratings[["ui", "mi", "rating"]].values
n_test = max(1, int(0.1 * len(obs)))
test_idx = rng.choice(len(obs), n_test, replace=False)
test_set = obs[test_idx]
train_mask = np.ones(len(obs), dtype=bool)
train_mask[test_idx] = False
train_set = obs[train_mask]

Rtr = np.full((n_users, n_items), 0.0)
Rtr[train_set[:, 0].astype(int), train_set[:, 1].astype(int)] = \
    train_set[:, 2] - col_mean[train_set[:, 1].astype(int)]

U, s, Vt = np.linalg.svd(Rtr, full_matrices=False)

# ---------- 4. 随机基线：随机抽 20 部电影，类型多样性熵的均值（满多样性） ----------
ent_baseline = []
for _ in range(1000):
    sample = rng.choice(n_items, TOPN, replace=False)
    p = genre_mat[sample].sum(axis=0)
    p = p / p.sum()
    ent_baseline.append(-np.sum(p[p > 0] * np.log2(p[p > 0])))
ent_random = float(np.mean(ent_baseline))

# ---------- 5. 茧房模拟：对 user#615 跑 25 轮 ----------
user = u_idx[USER_ID]
n_rated0 = int((ratings["ui"] == user).sum())
assert n_rated0 == 43, f"预期 user#{USER_ID} 看过 43 部，实际 {n_rated0}"

rated = set(ratings.loc[ratings["ui"] == user, "mi"].values)
user_vec = U[user] * s          # 口味画像 = U[:,user] ⊙ Σ
ent_hist = []
for rnd in range(N_ROUNDS):
    # 还没看过的候选电影
    cand_m = np.array([j for j in range(n_items) if j not in rated])
    # 用口味尺子量"最懂他"：候选电影潜向量 与 用户画像 的距离
    d = np.linalg.norm((Vt[:, cand_m] * s[:, None]) - user_vec[:, None], axis=0)
    top20 = cand_m[np.argsort(d)[:TOPN]]
    # 这个推荐列表里，类型分布有多多样 → 熵
    p = genre_mat[top20].sum(axis=0)
    p = p / p.sum()
    ent = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    ent_hist.append(ent)
    # 用户"看了最贴近的一部" → 画像被往已知方向推 → 下一轮重算
    nxt = cand_m[np.argmin(d)]
    rated.add(nxt)
    user_vec = user_vec + Vt[:, nxt] * s

ent_init = ent_hist[0]
ent_last10 = float(np.mean(ent_hist[-10:]))
narrow = (1 - ent_last10 / ent_random) * 100

print(f"\n=== 信息茧房账（user#{USER_ID}，初始已看 {n_rated0} 部）===")
print(f"  随机 20 部基线熵     = {ent_random:.2f} bit（满多样性）")
print(f"  第 1 轮推荐熵        = {ent_init:.2f} bit")
print(f"  末 10 轮平均熵       = {ent_last10:.2f} bit")
print(f"  相对基线收窄         = {narrow:.1f}%")

# ---------- 6. fig4：信息茧房熵随推荐轮次坍缩（单面板，Okabe-Ito 配色，去脊线） ----------
BLUE = "#0072B2"     # 基线
ORANGE = "#D55E00"   # 熵曲线
GREY = "#999999"

fig, ax = plt.subplots(figsize=(8, 5))
rounds = list(range(1, N_ROUNDS + 1))

# 熵曲线
ax.plot(rounds, ent_hist, "o-", color=ORANGE, lw=2, ms=5,
        label="每轮 Top-20 推荐列表的类型多样性熵")

# 随机基线（满多样性）
ax.axhline(ent_random, ls="--", color=BLUE, lw=1.6,
           label=f"随机挑 20 部基线 = {ent_random:.2f} bit")

# 末 10 轮平均线
ax.axhline(ent_last10, ls=":", color=ORANGE, lw=1.4, alpha=0.8,
           label=f"末 10 轮均值 = {ent_last10:.2f} bit")

# 标注末 10 轮区间
ax.axvspan(N_ROUNDS - 9.5, N_ROUNDS + 0.5, color=ORANGE, alpha=0.06)
ax.annotate(f"末 10 轮均值 {ent_last10:.2f} bit\n比随机基线收窄 {narrow:.1f}%",
            xy=(N_ROUNDS, ent_last10), xytext=(N_ROUNDS - 11, ent_last10 + 0.28),
            fontsize=10, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

ax.set_xlabel("推荐轮次（每轮让算法推 20 部「最懂他」的电影）")
ax.set_ylabel("推荐列表的类型多样性熵（bit）")
ax.set_title("信息茧房的数学签名：推荐越准，列表越窄")
ax.set_ylim(ent_last10 - 0.35, ent_random + 0.35)
ax.set_xlim(0.5, N_ROUNDS + 0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/fig4_cocoon_entropy.png")
plt.close(fig)
print(f"\nfig4 已写入 {OUT}/fig4_cocoon_entropy.png")
