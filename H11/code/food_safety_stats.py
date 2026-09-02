# -*- coding: utf-8 -*-
"""
H11 分析脚本：统计推断深化（置信区间 + 一类/二类错误 + 抽检实战）
数据集：国家市场监督管理总局《食品安全监督抽检不合格情况通报》官方数字
        （2024-12-29 至 2026-07-25 多期，逐期附来源 URL，零编造、官方可查）。
依赖：仅 numpy / matplotlib（受管 python venv）。
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 字体（中文无衬线，回退链）/ 配色 / 参数 ----------
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}

BASE = os.path.dirname(os.path.abspath(__file__))          # H11/code
ROOT = os.path.dirname(BASE)                                  # H11 根目录
CSV_PATH = os.path.join(ROOT, "data", "food_safety_bulletins.csv")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 读取官方通报数字 ----------
rows = []
with open(CSV_PATH, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rows.append({
            "date": r["通报日期"], "doc": r["文号"],
            "n": int(r["抽检批次"]), "bad": int(r["不合格批次"]),
            "url": r["来源URL"],
        })

N = sum(x["n"] for x in rows)
B = sum(x["bad"] for x in rows)
phat = B / N
print("=" * 64)
print(f"合并：抽检 {N} 批，不合格 {B} 批（2024-12-29 至 2026-07-25，四期）")
print(f"不合格率点估计  p̂ = {B}/{N} = {phat:.5f} = {phat*100:.2f}%")
print("=" * 64)

# ---------- Q2：95% 置信区间（Wald + Wilson）----------
z = 1.959963985
se = math.sqrt(phat * (1 - phat) / N)
lo, hi = phat - z * se, phat + z * se
print(f"\n[Q2] 标准误 SE = √(p̂(1-p̂)/n) = √({phat:.5f}×{1-phat:.5f}/{N}) = {se:.6f}")
print(f"95% Wald  区间 = {phat*100:.2f}% ± {z*se*100:.2f}% = [{lo*100:.2f}%, {hi*100:.2f}%]")
denom = 1 + z**2 / N
center = (phat + z**2 / (2 * N)) / denom
half = (z * math.sqrt(phat * (1 - phat) / N + z**2 / (4 * N**2))) / denom
print(f"95% Wilson 区间 = [{(center-half)*100:.2f}%, {(center+half)*100:.2f}%]（小比例更稳）")

# 逐期 95% CI（Wald），存盘供正文引用
per = []
for x in rows:
    r = x["bad"] / x["n"]
    se_i = math.sqrt(r * (1 - r) / x["n"])
    per.append((x["date"], r, r - z * se_i, r + z * se_i))
with open(os.path.join(ROOT, "data", "food_safety_rates.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["通报日期", "不合格率", "CI下", "CI上"])
    for d, r, l, h in per:
        w.writerow([d, f"{r:.5f}", f"{l:.5f}", f"{h:.5f}"])
    print(f"已保存 {os.path.join(ROOT, 'data', 'food_safety_rates.csv')}（逐期率与 95% CI）")

# ---------- Q1：四期是否稳定（卡方齐性检验，df=3）----------
chi2 = 0.0
for x in rows:
    exp_b = x["n"] * phat
    exp_g = x["n"] * (1 - phat)
    chi2 += (x["bad"] - exp_b) ** 2 / exp_b + ((x["n"] - x["bad"]) - exp_g) ** 2 / exp_g
crit = 7.815  # df=3, α=0.05
print(f"\n[Q1] 四期齐性卡方 χ² = {chi2:.3f}（df=3），临界值 7.815")
print("结论：四期不合格率高度稳定，无显著变化 —— 不是某期个例，是持续基线。" if chi2 < crit
      else "结论：四期存在显著差异。")

# ---------- Q4：最小样本量（以 80% 把握逮到≥1 批，二项检测概率）----------
def min_n_detect(p, power=0.80):
    return math.ceil(math.log(1 - power) / math.log(1 - p))
print("\n[Q4] 真实不合格率 p → 需抽多少批才有 80% 把握逮到≥1 批：")
for p in [0.001, 0.005, 0.01, phat]:
    n = min_n_detect(p)
    print(f"  p = {p*100:.1f}%  →  n ≥ {n} 批")

# 功效曲线数据（fig4）：P(逮到) = 1-(1-p)^n
p_rare = 0.001
ns = np.arange(1, 3001)
pdetect = 1 - (1 - p_rare) ** ns
n80 = min_n_detect(p_rare)

# ================= 绘图 =================
# fig1：四期不合格率 + 95% CI 误差棒
fig, ax = plt.subplots(figsize=(7.4, 4.3))
x = np.arange(len(per))
ys = [r * 100 for _, r, _, _ in per]
err = [(r - l) * 100 for _, r, l, _ in per]
ax.errorbar(x, ys, yerr=err, fmt="o", color=C["blue"], ecolor=C["orange"],
            elinewidth=2.2, capsize=6, ms=9, label="各期不合格率 ±95% CI")
ax.axhline(phat * 100, color=C["red"], ls="--", lw=1.8, label=f"四期合并 {phat*100:.2f}%")
ax.set_xticks(x); ax.set_xticklabels([d for d, _, _, _ in per], rotation=15, ha="right")
ax.set_ylabel("不合格率 (%)"); ax.set_ylim(0, 3.2)
ax.set_title("总局四期食品安全抽检：不合格率与 95% 置信区间")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig1_rates_ci.png")); plt.close(fig)

# fig2：点估计 vs 区间
fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.axhspan(lo * 100, hi * 100, color=C["orange"], alpha=0.20, label=f"95% 可信区间 [{lo*100:.2f}%, {hi*100:.2f}%]")
ax.plot([0], [phat * 100], "o", color=C["blue"], ms=12, label=f"点估计 {phat*100:.2f}%")
ax.set_xlim(-1, 1); ax.set_xticks([])
ax.set_ylabel("不合格率 (%)"); ax.set_ylim(0, 3.2)
ax.set_title("一个数 vs 一个范围：95% 区间告诉我们把握有多大")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig2_point_vs_interval.png")); plt.close(fig)

# fig3：一类/二类错误 2×2 判定表
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axis("off")
cells = [
    (0.04, 0.56, C["green"], "判合格 × 真实合格\n正确放行\n(绝大多数时候)"),
    (0.52, 0.56, C["red"], "判不合格 × 真实合格\n一类错误 α（假阳性）\n冤枉好厂、白挨罚"),
    (0.04, 0.06, C["orange"], "判合格 × 真实不合格\n二类错误 β（假阴性）\n毒菜上了桌、人吃进肚"),
    (0.52, 0.06, C["blue"], "判不合格 × 真实不合格\n正确拦截\n(统计功效 1−β)"),
]
for cx, cy, col, txt in cells:
    ax.add_patch(plt.Rectangle((cx, cy), 0.44, 0.34, facecolor=col, alpha=0.18,
                               edgecolor=col, lw=2))
    ax.text(cx + 0.22, cy + 0.17, txt, ha="center", va="center", fontsize=9,
            color=C["black"])
ax.text(0.27, 0.93, "真实情况 →", ha="center", fontsize=10)
ax.text(0.02, 0.78, "抽检\n判定 ↓", ha="left", fontsize=10)
ax.set_title("一类错误 vs 二类错误：天平两端，代价不同")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig3_errors_table.png")); plt.close(fig)

# fig4：功效曲线（样本量 vs 逮到概率）
fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.plot(ns, pdetect * 100, color=C["blue"], lw=2.4,
        label=f"真实不合格率 p={p_rare*100:.1f}% 时：抽 n 批逮到≥1 批的概率")
ax.axhline(80, color=C["red"], ls="--", lw=1.6, label="80% 把握线")
ax.axvline(n80, color=C["orange"], ls="--", lw=1.6, label=f"需 n≥{n80} 批才够")
ax.set_xlabel("抽检批次数 n"); ax.set_ylabel("逮到至少 1 批的概率 (%)")
ax.set_xlim(0, 3000); ax.set_ylim(0, 100)
ax.set_title("样本量越大，稀有无良添加越难漏网（功效曲线）")
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig4_power_curve.png")); plt.close(fig)

print("\n已生成 4 张图：figures/fig1_rates_ci.png .. fig4_power_curve.png")
print("完成。")
