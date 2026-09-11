"""M9 候选数据集 2（2023-01 快照）勘查"""
import pandas as pd

pd.set_option("display.width", 220)

P = "/tmp/p2.csv"
df = pd.read_csv(P, low_memory=False)
print("shape", df.shape)
print("\n===== 全部列名 =====")
print(list(df.columns))

print("\n===== 非空率 =====")
for c in df.columns:
    nn = df[c].notna().sum()
    if nn / len(df) > 0.05:
        print(f"{c:28s} {nn:6d}  {nn/len(df)*100:5.1f}%")

show = [c for c in ["Model", "Announced", "Status", "Display Size", "Weight",
                    "Internal Storage", "RAM", "Battery", "Price", "Chipset",
                    "Operating Software", "Display Type"] if c in df.columns]
print("\n===== 样例 =====")
for c in show:
    print(f"--- {c}")
    print(df[c].dropna().head(4).tolist())
