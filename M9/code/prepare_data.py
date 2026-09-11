"""
M9 数据准备：把 GSMArena 原始爬取表清洗成可聚类的数值特征表。

数据源：GitHub `AlbertHunduza/Smartphone-Project` 的 GSMArena.csv
（2023-01-04 从 gsmarena.com 爬取，11936 行 / 50 列，公开可下载）
URL: https://raw.githubusercontent.com/AlbertHunduza/Smartphone-Project/main/GSMArena.csv

输出：data/phone_clean.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "gsmarena_raw.csv"
OUT = ROOT / "data" / "phone_clean.csv"

# 原始快照不在本地时自动下载（公开仓库，不需要任何凭证）
SRC_URL = ("https://raw.githubusercontent.com/"
           "AlbertHunduza/Smartphone-Project/main/GSMArena.csv")
if not RAW.exists():
    import urllib.request
    RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"本地无原始数据，从公开仓库下载 ...\n  {SRC_URL}")
    urllib.request.urlretrieve(SRC_URL, RAW)
    print(f"已保存 {RAW}（{RAW.stat().st_size / 1024 / 1024:.1f} MB）")

df = pd.read_csv(RAW, low_memory=False)
print(f"原始：{df.shape[0]} 行 × {df.shape[1]} 列")


def clean(s):
    """去掉 GSMArena 文本里的窄空格与逗号分位符，便于正则。"""
    if not isinstance(s, str):
        return ""
    return s.replace("\u2009", " ").replace("\u202f", " ").replace(",", "")


def grab(pattern, s, scale=1.0):
    m = re.search(pattern, clean(s), re.I)
    if not m:
        return np.nan
    try:
        return float(m.group(1)) * scale
    except (ValueError, IndexError):
        return np.nan


out = pd.DataFrame()
out["model"] = df["Model"]
# Brands 形如 "Samsung\n1348 devices" —— 取第一行
out["brand"] = df["Brands"].astype(str).str.split("\n").str[0].str.strip()

# ---------- 年份 ----------
out["year"] = df["Announced"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
out.loc[out["year"] < 1994, "year"] = np.nan

# ---------- 价格（统一成欧元）----------
# Approx Price 90% 是 "About 110 EUR"。汇率按固定近似：1 USD=0.92 EUR，1 GBP=1.18 EUR
ap, pr = df["Approx Price"].astype(str), df["Price"].astype(str)
price = (
    ap.apply(lambda s: grab(r"([\d.]+)\s*EUR", s))
    .fillna(ap.apply(lambda s: grab(r"€\s*([\d.]+)", s)))
    .fillna(ap.apply(lambda s: grab(r"\$\s*([\d.]+)", s)) * 0.92)
    .fillna(pr.apply(lambda s: grab(r"([\d.]+)\s*EUR", s)))
    .fillna(pr.apply(lambda s: grab(r"€\s*([\d.]+)", s)))
    .fillna(pr.apply(lambda s: grab(r"\$\s*([\d.]+)", s)) * 0.92)
)
out["price_eur"] = price

# ---------- 规格 ----------
out["screen_in"] = df["Display Size"].apply(lambda s: grab(r"([\d.]+)\s*inch", s))
out["weight_g"] = df["Weight"].apply(lambda s: grab(r"([\d.]+)\s*g\b", s))
out["battery_mah"] = df["Battery"].apply(lambda s: grab(r"([\d.]+)\s*mAh", s))
out["thickness_mm"] = df["Dimensions"].apply(
    lambda s: grab(r"[\d.]+\s*x\s*[\d.]+\s*x\s*([\d.]+)", s)
)
out["camera_mp"] = df["Camera"].apply(lambda s: grab(r"([\d.]+)\s*MP", s))

# 分辨率 → 像素宽/高 → PPI
w = df["Display Resolution"].apply(lambda s: grab(r"(^\s*\d+)", s))
h = df["Display Resolution"].apply(lambda s: grab(r"x\s*(\d+)", s))
out["res_w"], out["res_h"] = w, h
out["ppi"] = np.sqrt(w ** 2 + h ** 2) / out["screen_in"]


def to_gb(v, u):
    v = float(v)
    return v * 1024 if u.upper() == "TB" else (v / 1024 if u.upper() == "MB" else v)


def split_storage(s):
    """Internal Storage 形如 '64GB 6GB RAM, 128GB 8GB RAM'，拆出存储与 RAM。"""
    if not isinstance(s, str):
        return np.nan, np.nan
    stor, ram = [], []
    for m in re.finditer(r"([\d.]+)\s*(TB|GB|MB)\s*([\d.]+)\s*(TB|GB|MB)?\s*RAM", s, re.I):
        stor.append(to_gb(m.group(1), m.group(2)))
        if m.group(3) and m.group(4):
            ram.append(to_gb(m.group(3), m.group(4)))
    return (max(stor) if stor else np.nan, max(ram) if ram else np.nan)


st = df["Internal Storage"].apply(split_storage)
out["storage_gb"] = [a for a, _ in st]
out["ram_gb"] = [b for _, b in st]

# ---------- 布尔特征 ----------
out["is_oled"] = df["Display Type"].astype(str).str.upper().str.contains("OLED").astype(float)
out["is_5g"] = df["Network Technology"].astype(str).str.upper().str.contains("5G").astype(float)
out["has_nfc"] = df["NFC"].astype(str).str.contains("Yes", case=False).astype(float)
out["has_jack"] = df["Headphone Jack"].astype(str).str.contains("Yes", case=False).astype(float)
out["chipset"] = df["Chipset"].where(df["Chipset"].notna(), "")

print(f"\n解析后：{out.shape}")
print("\n===== 各特征非空率 =====")
for c in out.columns:
    if c in ("model", "brand", "chipset"):
        continue
    nn = out[c].notna().sum()
    print(f"{c:14s} {nn:6d}  {nn/len(out)*100:5.1f}%")

# ---------- 筛选建模样本 ----------
# 先剔除平板：爬取表里混进了 iPad / Galaxy Tab 等平板（名字带关键词），
# 它们屏幕大、重量高，会污染"手机分档"的结论。折叠屏展开态虽也有 7.6–8 英寸，
# 但按型号名可以区分开，所以只按名字剔，不按尺寸砍。
TABLET_PAT = (r"iPad|Galaxy Tab|Tab Active|MediaPad|MatePad|Lenovo Tab|"
              r"ZenPad|Mi Pad|Nexus 7|Nexus 9|Tab\d|Pad [0-9]|"
              r"Galaxy Note 8\.0|Galaxy Note 10\.1|Galaxy Note Pro")
is_tablet = df["Model"].astype(str).str.contains(TABLET_PAT, case=False, regex=True)
out["is_tablet"] = is_tablet.astype(float)
print(f"\n按型号名识别为平板并剔除：{int(is_tablet.sum())} 款")
print(df.loc[is_tablet, "Model"].head(12).tolist())

NUM = ["price_eur", "screen_in", "weight_g", "battery_mah", "ppi",
       "storage_gb", "ram_gb", "camera_mp"]
mask = (
    (~is_tablet)
    & out["year"].between(2015, 2022)
    & out[NUM].notna().all(axis=1)
    & out["screen_in"].between(3.0, 8.5)
    & out["weight_g"].between(60, 450)
    & out["battery_mah"].between(1000, 12000)
    & out["price_eur"].between(30, 3000)
    & out["ppi"].between(100, 900)
    & out["camera_mp"].between(1, 250)
)
clean = out[mask].reset_index(drop=True)

print(f"\n===== 筛选后建模样本：{len(clean)} 行 / 全量 {len(out)} =====")
print(f"年份区间：{int(clean['year'].min())} – {int(clean['year'].max())}")
print("\n各字段分位：")
print(clean[NUM].describe().round(1).to_string())
print("\n品牌分布 Top15：")
print(clean["brand"].value_counts().head(15).to_string())

clean.to_csv(OUT, index=False)
print(f"\n已写出：{OUT}  形状 {clean.shape}")
print("列：" + ", ".join(clean.columns))
