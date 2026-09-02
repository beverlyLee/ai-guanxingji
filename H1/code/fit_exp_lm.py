#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳健 Levenberg-Marquardt 拟合 y = a*exp(c*x)（2 参数），避免手动 GN 发散。"""
import numpy as np

DIR = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/assets/h1-new-data"
CSV = f"{DIR}/gistemp_1880_2025.csv"
years, vals = [], []
with open(CSV) as f:
    f.readline()
    for line in f:
        line = line.strip()
        if not line:
            continue
        yr, v = line.split(",")
        years.append(int(yr)); vals.append(float(v))
years = np.array(years, float)
y = np.array(vals, float)
x = years - 1880.0
n = len(x)

def residuals(p, xd, yd):
    a, c = p
    return a * np.exp(c * xd) - yd

def jacobian(p, xd):
    a, c = p
    J = np.empty((len(xd), 2))
    J[:, 0] = np.exp(c * xd)
    J[:, 1] = a * xd * np.exp(c * xd)
    return J

def lm_fit(xd, yd, p0, max_iter=2000, lam0=1e-3):
    p = np.array(p0, float)
    lam = lam0
    r = residuals(p, xd, yd)
    cost = np.dot(r, r)
    for it in range(max_iter):
        J = jacobian(p, xd)
        H = J.T @ J
        g = J.T @ r
        diag = np.diag(H).copy()
        diag[diag == 0] = 1e-12
        dp = np.linalg.solve(H + lam * np.diag(diag), -g)
        p_new = p + dp
        r_new = residuals(p_new, xd, yd)
        cost_new = np.dot(r_new, r_new)
        if cost_new < cost:
            p, r, cost, lam = p_new, r_new, cost_new, max(lam / 3, 1e-8)
        else:
            lam = min(lam * 3, 1e6)
        if np.max(np.abs(dp)) < 1e-10:
            break
    return p, cost, it + 1

# 初始值：a=0.001, c=0.049
p, cost, iters = lm_fit(x, y, [0.001, 0.049])
a, c = p
yhat = a * np.exp(c * x)
r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
rss = np.sum((y - yhat) ** 2)
aicc = n * np.log(rss / n) + 2 * 2 + (2 * 2 * 3) / (n - 2 - 1)
print(f"指数 y=a*exp(c*x) LM 拟合 (iters={iters})")
print(f"a = {a:.6f}   c = {c:.5f}")
print(f"R² = {r2:.4f}   RSS = {rss:.5f}   AICc = {aicc:.2f}")
print(f"末值核对 2025(y={y[-1]:.2f}) fit={a*np.exp(c*x[-1]):.3f} | 1880(y={y[0]:.2f}) fit={a*np.exp(c*x[0]):.3f}")
print(f"RMS = {np.sqrt(np.mean((y-yhat)**2)):.4f}")
