"""M9 数据可用性勘查（一次性脚本，非最终实验代码）"""
import pandas as pd

pd.set_option("display.width", 200)

P = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_M9_无监督聚类_手机规格/data/gsmarena_raw_2021.csv"
df = pd.read_csv(P, low_memory=False)
print("shape", df.shape)

print("\n===== 候选字段非空率 =====")
for c in ["Brand", "Model Name", "Announced", "Price", "Internal", "RAM",
          "Weight", "Size", "Type_1", "Charging", "Chipset", "OS", "Status",
          "Resolution", "Display", "Camera", "Primary_camera", "Battery"]:
    if c in df.columns:
        nn = df[c].notna().sum()
        print(f"{c:20s} {nn:6d}  {nn/len(df)*100:5.1f}%")
    else:
        print(f"{c:20s} <不在列中>")

print("\n===== Price 原始样例（20 条）=====")
print(df["Price"].dropna().sample(20, random_state=1).tolist())

print("\n===== Internal 原始样例 =====")
print(df["Internal"].dropna().head(10).tolist())

print("\n===== Size(屏幕英寸) 原始样例 =====")
print(df["Size"].dropna().head(10).tolist())

print("\n===== Weight 原始样例 =====")
print(df["Weight"].dropna().head(10).tolist())

print("\n===== Announced 原始样例 =====")
print(df["Announced"].dropna().head(10).tolist())

print("\n===== 全部列名 =====")
print(list(df.columns))
