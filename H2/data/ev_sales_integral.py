#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2「积分」实跑脚本 —— 乘联会 NEV 月度零售销量（2018–2025，2020 缺）
三重积分对照，验证数据真能分析出东西：

核心概念：月销量（万辆/月）本身已是「速率 × 1 个月」的月聚合，
所以「累计销量 = 对月速率求积分」在离散世界就是【逐月求和】。

  ① 离散求和（矩形积分）= 累计：与乘联会官方年度零售对照，看误差
  ② 方法对照：若把月销量当「瞬时速率采样」做梯形/Simpson 数值积分，会怎样
     —— 揭示「数据到底是什么，决定用哪种积分」这个真实陷阱
  ③ 解析积分：复用 H1 拟合工具
     - 多项式（月度，AICc 选阶）→ 闭式原函数
     - 指数（年度总量，信号更干净）→ 闭式原函数
     解析积分 vs 数值积分，看光滑/季节项带来的偏差

所有数值打印供人工 QA，并 dump 到 results.json 给画图脚本用。
零编造红线：月度来自 CSV（券商研报汇编的乘联会零售），年度来自乘联会官方发布。
"""
import json
import numpy as np

DATA = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_H2_积分_NEV/data/cn_ev_monthly_2018_2025.csv"
OUT = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_H2_积分_NEV/data/results.json"

# ---------- 读取 CSV ----------
rows = []
with open(DATA, encoding="utf-8-sig") as f:
    f.readline()  # header
    for line in f:
        line = line.strip()
        if not line:
            continue
        y, m, v, src = line.split(",")
        rows.append((int(y), int(m), float(v), src))

years = sorted({r[0] for r in rows})
print("CSV 覆盖年份：", years, " 共", len(rows), "行月度数据")

t0_year, t0_month = 2018, 1
def to_t(y, m):
    return (y - t0_year) * 12 + (m - t0_month)

T = np.array([to_t(r[0], r[1]) for r in rows], dtype=float)
V = np.array([r[2] for r in rows], dtype=float)          # 万辆/月，速率
N = len(T)
print(f"全局样本 n={N}，t∈[{T[0]:.0f},{T[-1]:.0f}] 月")

# ============ ① 离散求和 = 积分本身（月度聚合求和） ============
# 矩形累计：截至第 k 月末，已过去月份的销量之和（每个月的销量=该月速率×1月）
cum_rect = np.cumsum(V)
# 梯形累计（把月销量当瞬时采样，piecewise-linear 近似）
cum_trap = np.zeros(N)
cum_trap[0] = V[0] / 2.0
for i in range(1, N):
    cum_trap[i] = cum_trap[i - 1] + (V[i - 1] + V[i]) / 2.0
print("\n=== ① 离散求和（矩形积分）= 累计销量（万辆） ===")
print(f"矩形累计(末)= {cum_rect[-1]:.1f}   梯形累计(末)= {cum_trap[-1]:.1f} 万辆")
print(f"（二者差 {(cum_trap[-1]-cum_rect[-1])/cum_rect[-1]*100:+.2f}%：梯形只因首尾月各算半截而略低）")

# ============ ② 逐年内「求和」vs 乘联会官方年度零售 + 方法对照 ============
OFFICIAL = {  # 乘联会官方年度零售（万辆）
    2018: 98.5, 2019: 100.0, 2020: 110.9,
    2021: 298.9, 2022: 567.4, 2023: 773.6, 2024: 1089.9, 2025: 1280.9,
}

def simpson_year(vals):  # 12 个等间距月度，复合 Simpson（n=12 偶数区间，h=1）
    h = 1.0
    s = vals[0] + vals[-1]
    s += 4 * np.sum(vals[1:-1:2])
    s += 2 * np.sum(vals[2:-1:2])
    return h / 3.0 * s

def trap_year(vals):  # 12 个月度，梯形
    return (vals[0] / 2 + np.sum(vals[1:-1]) + vals[-1] / 2)

print("\n=== ② 逐年内：求和(=积分) vs 官方年度零售；并附 梯形/Simpson 对照 ===")
print(f"{'年份':>5} {'求和(万)':>9} {'官方(万)':>9} {'求和误差%':>9} | {'梯形误%':>7} {'Simpson误%':>10}")
year_err = {}
for y in years:
    vals = np.array([r[2] for r in rows if r[0] == y], dtype=float)
    s = float(np.sum(vals))
    tr = trap_year(vals)
    sp = simpson_year(vals)
    off = OFFICIAL[y]
    e_sum, e_tr, e_sp = (s-off)/off*100, (tr-off)/off*100, (sp-off)/off*100
    year_err[y] = {"sum": round(s,1), "trap": round(tr,1), "simpson": round(sp,1),
                   "official": off, "err_sum": round(e_sum,2), "err_trap": round(e_tr,2), "err_simpson": round(e_sp,2)}
    print(f"{y:>5} {s:>9.1f} {off:>9.1f} {e_sum:>+9.2f} | {e_tr:>+7.2f} {e_sp:>+10.2f}")

num_total_7 = float(cum_rect[-1])
off_total_7 = sum(OFFICIAL[y] for y in years)        # 7 年官方（不含2020）
off_total_8 = sum(OFFICIAL.values())                 # 8 年官方（含2020）
print(f"\n7 年（不含2020）求和累计 = {num_total_7:.1f} 万  官方合计 = {off_total_7:.1f} 万  偏差 {(num_total_7-off_total_7)/off_total_7*100:+.2f}%")
print(f"8 年（含2020）官方零售合计 = {off_total_8:.1f} 万（我的累计缺2020约低 {OFFICIAL[2020]:.0f} 万）")

# ============ ③ 解析积分：复用 H1 拟合工具 ============
def aicc(rss, k, n):
    return n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)

print("\n=== ③-a 多项式拟合月度速率（t=月），AICc 选模型 ===")
poly_res = {}
for deg in [1, 2, 3, 4]:
    p = np.polyfit(T, V, deg)
    yhat = np.polyval(p, T)
    rss = np.sum((V - yhat) ** 2)
    r2 = 1 - rss / np.sum((V - V.mean()) ** 2)
    poly_res[deg] = {"poly": p.tolist(), "r2": float(r2), "aicc": float(aicc(rss, deg+1, N))}
    print(f"  deg={deg}  R²={r2:.4f}  AICc={poly_res[deg]['aicc']:.1f}")
best_deg = min(poly_res, key=lambda d: poly_res[d]["aicc"])
print(f"  → AICc 最优：deg={best_deg}")
p_best = np.array(poly_res[best_deg]["poly"])
int_coeff = [c / (len(p_best) - 1 - i + 1) for i, c in enumerate(p_best)] + [0.0]
int_poly = np.array(int_coeff)
analytic_poly = np.polyval(int_poly, T[-1]) - np.polyval(int_poly, T[0])
print(f"  deg={best_deg} 解析积分总量 = {analytic_poly:.1f} 万  vs 矩形累计 {cum_rect[-1]:.1f} 万  "
      f"偏差 {(analytic_poly-cum_rect[-1])/cum_rect[-1]*100:+.2f}%")

print("\n=== ③-b 指数拟合【年度总量】A·e^(k·t)，t=年-2018（信号干净，不做月度避免季节发散） ===")
yr_arr = np.array(sorted(OFFICIAL.keys()), dtype=float)          # 2018..2025
ann = np.array([OFFICIAL[y] for y in yr_arr], dtype=float)
t_ann = yr_arr - 2018.0                                          # 居中，避免 exp 溢出
# 闭式：ln y = lnA + k·t  → 对 (t, ln y) 做线性最小二乘，稳定不发散
slope, intercept = np.polyfit(t_ann, np.log(ann), 1)
k, A = float(slope), float(np.exp(intercept))
yhat_a = A * np.exp(k * t_ann)
r2_a = 1 - np.sum((ann - yhat_a) ** 2) / np.sum((ann - ann.mean()) ** 2)
analytic_exp = (A / k) * (np.exp(k * t_ann[-1]) - np.exp(k * t_ann[0]))
print(f"  A={A:.3f}  k={k:.4f}  R²={r2_a:.4f}")
print(f"  指数解析积分（年度总量积分）= {analytic_exp:.1f} 万  vs 官方 8 年合计 {off_total_8:.1f} 万  "
      f"偏差 {(analytic_exp-off_total_8)/off_total_8*100:+.2f}%")
# 年度拟合值也存下来画图
exp_fit_annual = {int(yr_arr[i]): round(float(yhat_a[i]), 1) for i in range(len(yr_arr))}

# ============ 累计里程碑（纯积分产物的叙事点） ============
print("\n=== 累计里程碑（从 2018-01 起，矩形累计；注：2020 缺故约低 111 万） ===")
milestones = [100, 500, 1000, 2000, 3000]
mi = 0
milestone_hits = {}
for i in range(N):
    if mi < len(milestones) and cum_rect[i] >= milestones[mi]:
        tot_m = int(T[i]); yr = t0_year + tot_m // 12; mo = t0_month + tot_m % 12
        if mo > 12: yr += 1; mo -= 12
        print(f"  累计破 {milestones[mi]} 万 → 约 {yr}-{mo:02d}（t={T[i]:.0f}月, 累计{cum_rect[i]:.0f}万）")
        milestone_hits[milestones[mi]] = f"{yr}-{mo:02d}"
        mi += 1

# ============ 输出 JSON ============
results = {
    "years": years, "T": T.tolist(), "V": V.tolist(),
    "cum_rect": cum_rect.tolist(), "cum_trap": cum_trap.tolist(),
    "year_err": year_err,
    "num_total_7": num_total_7, "off_total_7": off_total_7, "off_total_8": off_total_8,
    "poly_res": {str(k): v for k, v in poly_res.items()}, "best_deg": best_deg,
    "analytic_total_poly": float(analytic_poly),
    "exp": {"A": float(A), "k": float(k), "r2": float(r2_a), "analytic_total": float(analytic_exp),
            "fit_annual": exp_fit_annual},
    "milestones": milestone_hits, "official": OFFICIAL,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n✅ 结果已写入 {OUT}")
