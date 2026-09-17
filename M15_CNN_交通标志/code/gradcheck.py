#!/usr/bin/env python3
"""
M15 梯度校验：用数值梯度（中心差分）验证手写卷积 / BatchNorm / 池化 / 全连接的
反向传播实现是否正确。任何一层写错，训练出来的准确率都是废数据。

判据用规范化的相对误差 ||g_ana - g_num|| / (||g_ana|| + ||g_num||)，
比逐元素最大相对误差稳健（后者遇到本身接近 0 的分量会假报警）。

一个真实现象：卷积层偏置后面紧跟 BatchNorm 时，真梯度恒为 0。
因为 BN 会减掉通道均值，给整层加一个常数偏移会被均值原样消掉，
所以这一项解析梯度 ≈ 0 是正确结果，不是 bug。

用法：python code/gradcheck.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import (BatchNorm, Conv2, Flatten, Linear, MaxPool2,  # noqa: E402
                       ReLU, softmax_ce)

EPS = 1e-5  # float64 下中心差分的经验取值（太大截断误差涨，太小舍入噪声涨）
DT = np.float64


def make(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4, 3, 4, 4)).astype(DT)
    y = np.array([0, 1, 2, 1], dtype=np.int64)
    ops = [
        Conv2(3, 2, 3, 1, rng, dtype=DT),
        BatchNorm(2, dtype=DT),
        ReLU(),
        MaxPool2(),
        Flatten(),
        Linear(8, 3, rng, dtype=DT),
    ]
    return x, y, ops


def loss_and_grads(x, y, ops):
    for o in ops:
        if isinstance(o, BatchNorm):
            o.training = True
    h = x
    for o in ops:
        h = o.forward(h)
    loss, d, _ = softmax_ce(h, y)
    for o in reversed(ops):
        d = o.backward(d)
    return loss


def numeric_grad(x, y, ops, param):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = param[i]
        param[i] = old + EPS
        lp = loss_and_grads(x, y, ops)
        param[i] = old - EPS
        lm = loss_and_grads(x, y, ops)
        param[i] = old
        grad[i] = (lp - lm) / (2 * EPS)
        it.iternext()
    return grad


def verdict(g_ana, g_num):
    na, nb = np.max(np.abs(g_ana)), np.max(np.abs(g_num))
    denom = np.linalg.norm(g_ana) + np.linalg.norm(g_num)
    err = float(np.linalg.norm(g_ana - g_num) / denom) if denom > 1e-12 else 0.0
    if na < 1e-6 and nb < 1e-4:
        return "PASS", err, "真梯度为 0（该参数后面紧跟 BN，常数偏移被均值消掉），数值项只是差分噪声"
    if err < 1e-5:
        return "PASS", err, ""
    if err < 1e-3:
        return "WARN", err, "接近阈值，可能是差分噪声"
    return "FAIL", err, "反向传播可能有错"


def main():
    x, y, ops = make()
    c1, b1, f1 = ops[0], ops[1], ops[5]

    loss = loss_and_grads(x, y, ops)
    print(f"loss = {loss:.6f}\n")

    checks = [
        ("Conv2.W", c1.W, c1.dW),
        ("Conv2.b", c1.b, c1.db),
        ("BatchNorm.gamma", b1.g, b1.dg),
        ("BatchNorm.beta", b1.b, b1.db),
        ("Linear.W", f1.W, f1.dW),
        ("Linear.b", f1.b, f1.db),
    ]

    ok = True
    for name, p, g_ana in checks:
        g_num = numeric_grad(x, y, ops, p)
        flag, e, note = verdict(g_ana, g_num)
        if flag == "FAIL":
            ok = False
        line = f"[{flag}] {name:16s} 相对误差 = {e:.3e}  (解析 max|g|={np.max(np.abs(g_ana)):.4g}, 数值 max|g|={np.max(np.abs(g_num)):.4g})"
        print(line)
        if note:
            print(f"       说明：{note}")

    print()
    if ok:
        print("全部通过：反向传播实现正确，可以开始训练。")
    else:
        print("存在 FAIL：请先修反向传播，不要跑训练。")
        sys.exit(1)


if __name__ == "__main__":
    main()
