# -*- coding: utf-8 -*-
"""口径交叉校验：cma_cat 官方等级 vs cma_wind 换算等级（🏃 可复现）"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "data", "typhoon_storms_1949_2024.csv")

rows = list(csv.DictReader(open(P, encoding="utf-8")))
print(f"读入 {len(rows)} 条")

V_CUT = [10.8, 17.2, 24.5, 32.7, 41.5, 51.0]
CATS = ["热带低压", "热带风暴", "强热带风暴", "台风", "强台风", "超强台风"]


def f2(x):
    return float(x) if x not in ("", None) else np.nan


def cat_from_v(v):
    if not np.isfinite(v) or v < V_CUT[0]:
        return -1
    for j in range(len(CATS)):
        if v < V_CUT[j]:
            return j - 1
    return len(CATS) - 1


cat = np.array([int(r["cma_cat"]) for r in rows])
vmax = np.array([f2(r["vmax_ms"]) for r in rows])
year = np.array([int(r["year"]) for r in rows])
name = np.array([r["name"] for r in rows])
hit = np.array([r["landfall_cn"] == "1" for r in rows])

cat_v = np.array([cat_from_v(v) for v in vmax])
both = (cat >= 0) & (cat_v >= 0)
print(f"\ncma_cat 与按风速判级同时可得的样本 = {int(both.sum())}")
print("混淆矩阵（行=cma_cat 官方，列=cma_wind 换算）：")
print("        " + "".join(f"{c:>8s}" for c in CATS))
agree = 0
for i in range(6):
    line = f"{CATS[i]:>6s}  "
    for j in range(6):
        n = int(((cat == i) & (cat_v == j)).sum())
        line += f"{n:>8d}"
        if i == j:
            agree += n
    print(line)
print(f"完全一致率 = {agree/max(int(both.sum()),1)*100:.2f}%")

print("\ncma_cat=5（超强台风）的 vmax 分布：")
v5 = vmax[(cat == 5) & np.isfinite(vmax)]
print(f"  有风速的 {len(v5)} / 全部 {int((cat==5).sum())} 个")
if len(v5):
    print(f"  分位数 5/25/50/75/95% = "
          f"{np.percentile(v5,5):.1f} / {np.percentile(v5,25):.1f} / {np.percentile(v5,50):.1f} / "
          f"{np.percentile(v5,75):.1f} / {np.percentile(v5,95):.1f} m/s")
    print(f"  vmax ≥ 51 m/s 的比例 = {(v5>=51).mean()*100:.1f}%")

print("\n按年代看超强台风(cma_cat=5)占全部的比例：")
for d0 in range(1940, 2030, 10):
    m = (year >= d0) & (year < d0 + 10)
    if m.sum() == 0:
        continue
    print(f"  {d0}s: 总数 {int(m.sum()):4d}  超强 {int((cat[m]==5).sum()):4d}  "
          f"占比 {(cat[m]==5).mean()*100:5.1f}%   年均超强 {(cat[m]==5).sum()/ (m.sum()/ (m.sum() and 1)) / 10:.2f}")

print("\n抽查几个著名台风（年份 + 名称模糊匹配）：")
for key in ["HAIYAN", "MERANTI", "USAGI", "SANBA", "NEPARTAK", "MANGKHUT", "LEKIMA", "BEBINCA"]:
    m = np.array([key in n.upper() for n in name])
    if m.any():
        i = np.where(m)[0]
        for j in i[:2]:
            print(f"  {year[j]} {name[j]:12s} cat={cat[j]} vmax={vmax[j]:.1f} m/s 登陆中国={hit[j]}")

print("\n登陆中国的台风中，各等级占比（检验强台风/超强是否过多）：")
for i in range(6):
    n_h = int(((cat == i) & hit).sum())
    if n_h:
        print(f"  {CATS[i]:6s} 登陆 {n_h:4d} 个，占登陆总数 {n_h/int(hit.sum())*100:5.1f}%")
print(f"  登陆总数 = {int(hit.sum())}")
