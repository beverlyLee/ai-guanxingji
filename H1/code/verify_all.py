#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NASA GISTEMP 1880-2025 数值锁定脚本（H1 重写唯一可信数值源）
重跑：直线 / 二次 / 指数 三模型；R²、AICc；梯度下降收敛验证；二次导数判加速。
所有数值打印出来供人工 QA。
"""
import numpy as np

DIR = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/assets/h1-new-data"
CSV = f"{DIR}/gistemp_1880_2025.csv"

# ---------- 读取 ----------
years, vals = [], []
with open(CSV) as f:
    f.readline()  # header
    for line in f:
        line = line.strip()
        if not line:
            continue
        yr, v = line.split(",")
        years.append(int(yr))
        vals.append(float(v))
years = np.array(years)
vals = np.array(vals)
n = len(years)
print(f"n = {n}  年份范围 {years[0]}-{years[-1]}")

x = years.astype(float) - 1880.0          # 距 1880 的年数, 0..145
y = vals
xx = x

def r2_and_mse(poly, xin, yin):
    yhat = np.polyval(poly, xin)
    ss_res = np.sum((yin - yhat) ** 2)
    ss_tot = np.sum((yin - np.mean(yin)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mse = ss_res / len(yin)
    return r2, mse

def aicc(rss, k, n):
    # k = 参数个数
    return n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)

# ================= 直线 y = w*x + b =================
A = np.vstack([x, np.ones_like(x)]).T
(w, b), res, _, _ = np.linalg.lstsq(A, y, rcond=None)
lin_poly = np.array([w, b])          # 高次在前
r2_lin, mse_lin = r2_and_mse(lin_poly, x, y)
rss_lin = np.sum((y - np.polyval(lin_poly, x)) ** 2)
aicc_lin = aicc(rss_lin, 2, n)
print("\n=== 直线 y = w*x + b ===")
print(f"w = {w:.5f} °C/yr   b = {b:.5f}  (意味着 1880 外推值)")
print(f"R² = {r2_lin:.4f}   MSE = {mse_lin:.5f}   AICc = {aicc_lin:.2f}")

# ================= 二次 y = a*x² + b*x + c =================
A2 = np.vstack([x**2, x, np.ones_like(x)]).T
(a2, b2, c2), res2, _, _ = np.linalg.lstsq(A2, y, rcond=None)
quad_poly = np.array([a2, b2, c2])
r2_q, mse_q = r2_and_mse(quad_poly, x, y)
rss_q = np.sum((y - np.polyval(quad_poly, x)) ** 2)
aicc_q = aicc(rss_q, 3, n)
print("\n=== 二次 y = a*x² + b*x + c (x=yr-1880) ===")
print(f"a = {a2:.7f}   b = {b2:.5f}   c = {c2:.5f}")
print(f"R² = {r2_q:.4f}   MSE = {mse_q:.5f}   AICc = {aicc_q:.2f}")

# ================= 指数 y = a*exp(c*x) (Gauss-Newton 非线性) =================
def fit_exp(xd, yd, iters=20000, lr=0.001, a0=0.001, c0=0.05):
    a, c = a0, c0
    for _ in range(iters):
        yh = a * np.exp(c * xd)
        e = yh - yd
        da = np.sum(e * np.exp(c * xd)) / len(xd)
        dc = np.sum(e * a * xd * np.exp(c * xd)) / len(xd)
        a -= lr * da
        c -= lr * dc
    return a, c
a_e, c_e = fit_exp(x, y)
exp_poly = (a_e, c_e)
yhat_e = a_e * np.exp(c_e * x)
r2_e = 1 - np.sum((y - yhat_e) ** 2) / np.sum((y - np.mean(y)) ** 2)
rss_e = np.sum((y - yhat_e) ** 2)
aicc_e = aicc(rss_e, 2, n)
print("\n=== 指数 y = a*exp(c*x) (Gauss-Newton) ===")
print(f"a = {a_e:.6f}   c = {c_e:.5f}")
print(f"R² = {r2_e:.4f}   MSE = {rss_e/n:.5f}   AICc = {aicc_e:.2f}")

# ================= 梯度下降收敛验证（含特征缩放） =================
print("\n=== 梯度下降：特征缩放前后对比 ===")
# 不缩放: x 0..145
w1, b1 = 0.0, 0.0
eta = 0.01
try:
    for _ in range(20000):
        yh = w1 * x + b1
        e = yh - y
        dw = (2 / n) * np.sum(e * x)
        db = (2 / n) * np.sum(e)
        w1 -= eta * dw
        b1 -= eta * db
    print(f"[不缩放 x=yr-1880] 末值 w={w1:.4f} b={b1:.4f}  (期望 ~0.00829 / -0.51886)")
except Exception as ex:
    print(f"[不缩放 x=yr-1880] 发散: {ex}")

# 缩放: t=(yr-1880)/10
t = (years - 1880.0) / 10.0
w2, b2g = 0.0, 0.0
for _ in range(20000):
    yh = w2 * t + b2g
    e = yh - y
    dw = (2 / n) * np.sum(e * t)
    db = (2 / n) * np.sum(e)
    w2 -= eta * dw
    b2g -= eta * db
print(f"[缩放 t=(yr-1880)/10] w_t={w2:.6f} b={b2g:.5f}  -> 真实w=w_t/10={w2/10:.5f} b={b2g:.5f}")
print(f"闭式 w=0.00829 对比 GD w/10={w2/10:.5f}  闭式 b=-0.51886 对比 GD b={b2g:.5f}")

# ================= 二次导数：判加速 / 到顶 =================
print("\n=== 二次导数 dy/dx = 2a*t + b （升温速率），2a=加速度 ===")
print(f"加速度 2a = {2*a2:.7f} °C/yr²  (正 => 升温在加速)")
for yr in [1880, 1950, 1980, 2000, 2025]:
    tt = yr - 1880
    slope = 2 * a2 * tt + b2
    print(f"  {yr}: 斜率 dy/dx = {slope:.4f} °C/yr")
# 是否到顶：斜率=0 的年份（若有且落在区间内）
if 2 * a2 != 0:
    t_top = -b2 / (2 * a2)
    yr_top = 1880 + t_top
    print(f"斜率=0 的年份 ≈ {yr_top:.1f}（区间外/或单调递增检查）")

# ================= 残差对比（模型选择硬证据） =================
print("\n=== 残差 RMS 对比 ===")
print(f"直线 RMS = {np.sqrt(np.mean((y-np.polyval(lin_poly,x))**2)):.4f}")
print(f"二次 RMS = {np.sqrt(np.mean((y-np.polyval(quad_poly,x))**2)):.4f}")
print(f"指数 RMS = {np.sqrt(np.mean((y-yhat_e)**2)):.4f}")

# ================= 关键年份原始值核对 =================
print("\n=== 末 6 年原始值核对 ===")
for i in range(-6, 0):
    print(f"  {years[i]}: {y[i]:.2f}")
