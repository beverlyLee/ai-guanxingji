# -*- coding: utf-8 -*-
"""探查 IBTrACS NetCDF 的变量结构（🏃 可复现）"""
import os
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
NC = os.path.join(HERE, "data", "IBTrACS.WP.v04r01.nc")

ds = netCDF4.Dataset(NC)

print("=== dimensions ===")
for k, v in ds.dimensions.items():
    print(f"  {k:14s} {len(v)}")

print("\n=== variables ===")
for name, var in ds.variables.items():
    print(f"  {name:18s} {str(var.dimensions):28s} {var.dtype}")

print("\n=== key variable attributes ===")
keys = [
    "sid", "season", "number", "basin", "subbasin", "name",
    "time", "lat", "lon", "wind", "pres", "dist2land",
    "cma_wind", "cma_pres", "usa_wind", "usa_pres", "wmo_wind", "wmo_pres",
    "nature", "track_type", "landfall", "usa_sshs",
]
for key in keys:
    if key in ds.variables:
        v = ds.variables[key]
        attrs = {a: v.getncattr(a) for a in v.ncattrs()
                 if a in ("units", "long_name", "_FillValue", "standard_name")}
        print(f"  {key:12s} {str(v.dimensions):26s} {attrs}")

# 抽样看数据形态
print("\n=== sample: first 5 storms ===")
n = ds.dimensions["storm"].size if "storm" in ds.dimensions else 0
print("n_storms =", n)
if "season" in ds.variables:
    season = ds.variables["season"][:]
    print("season range:", int(season.min()), "-", int(season.max()))
if "name" in ds.variables:
    names = ds.variables["name"][:5]
    for nm in names:
        print("   name raw:", nm)

ds.close()
