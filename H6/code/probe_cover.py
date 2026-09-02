# -*- coding: utf-8 -*-
"""探查各机构数据覆盖率与登陆判定口径（🏃 可复现）"""
import os
import numpy as np
import netCDF4
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
NC = os.path.join(HERE, "data", "IBTrACS.WP.v04r01.nc")
ds = netCDF4.Dataset(NC)

exec(open(os.path.join(HERE, "china_poly.py"), encoding="utf-8").read())

Y0, Y1 = 1949, 2024


def get(name):
    """读数值变量：先转 float 再填 NaN（int16 的 masked array 不能直接填 NaN）"""
    if name not in ds.variables:
        return None
    a = ds.variables[name][:]
    if isinstance(a, np.ma.MaskedArray):
        a = np.ma.filled(a.astype(float), np.nan)
    return np.asarray(a, dtype=float)


season = get("season").astype(float)
season = np.where(season <= 0, np.nan, season)
lat = get("lat").astype(float)
lon = get("lon").astype(float)
lat = np.where(np.abs(lat) > 900, np.nan, lat)
lon = np.where(np.abs(lon) > 900, np.nan, lon)

sel = np.isfinite(season) & (season >= Y0) & (season <= Y1)
print(f"时段 {Y0}–{Y1} 风暴数 = {int(sel.sum())} / 全部 {len(season)}")

for v in ["cma_wind", "cma_pres", "cma_cat", "wmo_wind", "wmo_pres",
          "usa_wind", "usa_pres", "tokyo_wind", "dist2land", "landfall"]:
    a = get(v)
    if a is None:
        print(f"  {v:12s} 不存在")
        continue
    a = a.astype(float)
    a = np.where(a <= -900, np.nan, a)
    sub = a[sel]
    valid = np.isfinite(sub) & (sub != 0) if v.endswith(("wind", "pres")) else np.isfinite(sub)
    print(f"  {v:12s} 时段内有效点覆盖率 = {valid.mean()*100:6.2f}%   "
          f"非缺失风暴数 = {int(valid.any(axis=1).sum())}")

# cma_cat 取值分布
cma_cat = get("cma_cat").astype(float)
cma_cat = np.where(cma_cat <= -900, np.nan, cma_cat)
sub = cma_cat[sel]
vals, cnts = np.unique(sub[np.isfinite(sub)], return_counts=True)
print(f"\ncma_cat 取值分布：{dict(zip(vals.astype(int).tolist(), cnts.tolist()))}")

# landfall / dist2land 取值分布
for v in ["landfall", "dist2land"]:
    a = get(v).astype(float)
    a = np.where(a <= -900, np.nan, a)
    sub = a[sel]
    z = sub[np.isfinite(sub)]
    print(f"{v}: 最小 {z.min()}, ==0 的点数 {int((z==0).sum())}, "
          f"<=5km {int((z<=5).sum())}, 总有效点 {len(z)}")

# 登陆中国判定对比
def in_china(la, lo):
    ok = np.isfinite(la) & np.isfinite(lo)
    res = np.zeros(len(la), dtype=bool)
    pts = np.column_stack([np.where(ok, lo, 0.0), np.where(ok, la, 0.0)])
    for p in CN_PATHS:
        res |= p.contains_points(pts)
    return res & ok


lf = get("landfall").astype(float)
lf = np.where(lf <= -900, np.nan, lf)
d2l = get("dist2land").astype(float)
d2l = np.where(d2l <= -900, np.nan, d2l)

lat_s, lon_s, lf_s, d2l_s = lat[sel], lon[sel], lf[sel], d2l[sel]
cn = np.zeros_like(lat_s, dtype=bool)
for i in range(lat_s.shape[0]):
    cn[i] = in_china(lat_s[i], lon_s[i])

print("\n登陆中国判定对比（1949–2024）：")
for name, cond in [
    ("dist2land==0 且在中国", (d2l_s == 0) & cn),
    ("landfall==0 且在中国", (lf_s == 0) & cn),
    ("dist2land<=25km 且在中国", (d2l_s <= 25) & cn),
    ("landfall<=25km 且在中国", (lf_s <= 25) & cn),
]:
    storms = int(cond.any(axis=1).sum())
    print(f"  {name:26s} 风暴数 = {storms:5d}  年均 {storms/(Y1-Y0+1):.2f}")

ds.close()
