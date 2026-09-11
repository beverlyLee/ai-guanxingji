"""建模样本侦查：折叠屏 / iPhone / 极端机型是否真的在样本内（决定叙事能否兑现）"""
import numpy as np
import pandas as pd

P = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_M9_无监督聚类_手机规格/data/phone_clean.csv"
d = pd.read_csv(P)

print(f"样本 {len(d)} 行")
print("\n===== 折叠屏候选机型 =====")
pat = "Fold|Flip|Razr|Mate X|Mix Fold|ZenFone Fold|W20|W21|Z Fold|Z Flip|V Fold"
f = d[d["model"].str.contains(pat, case=False, na=False)]
print(f"命中 {len(f)} 款：")
print(f[["model", "year", "price_eur", "screen_in", "weight_g", "battery_mah", "ram_gb"]]
      .to_string(index=False))

print("\n===== Apple 机型 =====")
a = d[d["brand"] == "Apple"]
print(f"命中 {len(a)} 款")
print(a[["model", "year", "price_eur", "screen_in", "weight_g", "battery_mah", "ppi", "ram_gb"]]
      .to_string(index=False))

print("\n===== 价格最贵的 12 款 =====")
print(d.nlargest(12, "price_eur")[["model", "brand", "year", "price_eur", "screen_in", "battery_mah"]]
      .to_string(index=False))

print("\n===== 电池最大的 10 款 =====")
print(d.nlargest(10, "battery_mah")[["model", "brand", "year", "price_eur", "battery_mah", "weight_g"]]
      .to_string(index=False))

print("\n===== 重量最大的 10 款 =====")
print(d.nlargest(10, "weight_g")[["model", "brand", "year", "price_eur", "weight_g", "screen_in"]]
      .to_string(index=False))

print("\n===== 存储最小的 10 款（看是否混入功能机）=====")
print(d.nsmallest(10, "storage_gb")[["model", "brand", "year", "price_eur", "storage_gb", "ram_gb", "screen_in"]]
      .to_string(index=False))

print("\n===== 特征相关性（log 后）=====")
num = ["price_eur", "screen_in", "weight_g", "battery_mah", "ppi", "storage_gb", "ram_gb", "camera_mp"]
lg = np.log1p(d[num])
print(lg.corr().round(2).to_string())
