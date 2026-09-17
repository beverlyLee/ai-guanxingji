#!/usr/bin/env python3
"""
M15 实验：纯 numpy 从零手写卷积网络，在 GTSRB（43 类交通标志）上跑对照。

网络结构（输入 3x32x32，逐段堆 Conv-[BN]-ReLU-Pool2）
    Conv 3->16, 3x3, pad=1     -> 16x32x32
    [BatchNorm]  ReLU  MaxPool2 -> 16x16x16
    Conv 16->32, 3x3, pad=1    -> 32x16x16
    [BatchNorm]  ReLU  MaxPool2 -> 32x8x8
    Conv 32->64, 3x3, pad=1    -> 64x8x8
    [BatchNorm]  ReLU  MaxPool2 -> 64x4x4
    Flatten -> 1024
    Linear 1024->64  ReLU
    Linear 64->43  Softmax + 交叉熵

两个阶段
    阶段一 学习率扫描：4000 张训练子样本、800 张验证子样本、6 轮，
            扫 use_bn ∈ {True, False} x lr ∈ {0.005, 0.01, 0.05, 0.1, 0.3}
    阶段二 正式对照：全量训练集、12 轮，三组
            A 带 BN    @ 扫描出的 BN 最优学习率
            B 不带 BN  @ 与 A 完全相同的学习率（隔离出 BN 单独的贡献）
            C 不带 BN  @ 扫描出的无 BN 最优学习率（双方各用自己最好的设置）

产出
    data/stats.json    文章所有数字的唯一来源
    data/plot_data.npz 画图用的原始曲线与混淆矩阵
    data/run.log       运行日志
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NPZ = DATA / "gtsrb32.npz"
STATS = DATA / "stats.json"
PLOT = DATA / "plot_data.npz"
LOG = DATA / "run.log"

SEED = 20260917
N_CLASS = 43
IMG = 32
EPOCHS = 12
BATCH = 64
MOMENTUM = 0.9
CHANNELS = (16, 32, 64)

# 阶段一 学习率扫描的规模（纯 numpy 跑 CPU，控制在这一档才能几分钟出结果）
SCAN_LRS = (0.005, 0.01, 0.05, 0.1, 0.3)
SCAN_EPOCHS = 6
SCAN_N_TRAIN = 4000
SCAN_N_VAL = 800
# 阶段二 放大学习率的那一档
LR_FINAL_BIG = 0.05

CLASS_NAMES = [
    "限速20", "限速30", "限速50", "限速60", "限速70", "限速80", "解除限速80",
    "限速100", "限速120", "禁止超车", "禁止货车超车", "路口优先", "干道优先",
    "让行", "停车", "禁止机动车", "禁止货车", "禁止驶入", "注意危险", "左急弯",
    "右急弯", "连续弯道", "路面不平", "路面湿滑", "右侧变窄", "施工", "信号灯",
    "行人", "儿童过街", "自行车过街", "注意冰雪", "野生动物", "解除全部限制",
    "前方右转", "前方左转", "只准直行", "直行或右转", "直行或左转", "靠右行驶",
    "靠左行驶", "环岛", "解除禁止超车", "解除货车超车",
]


# ---------------------------------------------------------------- 日志
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


# ---------------------------------------------------------------- 基础算子
def im2col(x, k, pad):
    """(N,C,H,W) -> (N, C*k*k, H*W)，索引顺序 (i*k+j)*C+c"""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    cols = np.empty((N, C * k * k, H * W), dtype=x.dtype)
    idx = 0
    for i in range(k):
        for j in range(k):
            cols[:, idx * C:(idx + 1) * C, :] = xp[:, :, i:i + H, j:j + W].reshape(N, C, H * W)
            idx += 1
    return cols


def col2im(cols, shape, k, pad):
    N, C, H, W = shape
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    idx = 0
    for i in range(k):
        for j in range(k):
            xp[:, :, i:i + H, j:j + W] += cols[:, idx * C:(idx + 1) * C, :].reshape(N, C, H, W)
            idx += 1
    return xp[:, :, pad:pad + H, pad:pad + W] if pad else xp


class Conv2:
    def __init__(self, cin, cout, k, pad, rng, dtype=np.float32):
        self.k, self.pad, self.cin, self.cout = k, pad, cin, cout
        std = np.sqrt(2.0 / (cin * k * k))
        self.W = rng.normal(0, std, (cout, k, k, cin)).astype(dtype)
        self.b = np.zeros(cout, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def pairs(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, x):
        self.shape = x.shape
        self.cols = im2col(x, self.k, self.pad)
        Wf = self.W.reshape(self.cout, -1)
        out = np.matmul(Wf, self.cols)
        out += self.b[None, :, None]
        return out.reshape(x.shape[0], self.cout, x.shape[2], x.shape[3])

    def backward(self, d):
        N = self.shape[0]
        dr = d.reshape(N, self.cout, -1)
        Wf = self.W.reshape(self.cout, -1)
        self.dW[...] = np.matmul(dr, self.cols.transpose(0, 2, 1)).sum(axis=0).reshape(self.dW.shape)
        self.db[...] = dr.sum(axis=(0, 2))
        dcols = np.matmul(Wf.T, dr)
        return col2im(dcols, self.shape, self.k, self.pad)


class BatchNorm:
    """通道维标准化。4D 输入按 (N,H,W) 统计，2D 输入按 N 统计。"""

    def __init__(self, C, momentum=0.9, eps=1e-5, dtype=np.float32):
        self.C, self.momentum, self.eps = C, momentum, eps
        self.g = np.ones(C, dtype=dtype)
        self.b = np.zeros(C, dtype=dtype)
        self.rm = np.zeros(C, dtype=dtype)
        self.rv = np.ones(C, dtype=dtype)
        self.dg = np.zeros(C, dtype=dtype)
        self.db = np.zeros(C, dtype=dtype)
        self.training = True

    def pairs(self):
        return [(self.g, self.dg), (self.b, self.db)]

    def n_params(self):
        return 2 * self.C

    def _shape(self, ndim):
        return (1, self.C) + (1,) * (ndim - 2)

    def forward(self, x):
        self.x = x
        shp = self._shape(x.ndim)
        axes = (0,) + tuple(range(2, x.ndim))
        if self.training:
            self.mu = x.mean(axis=axes, keepdims=True)
            self.var = x.var(axis=axes, keepdims=True)
            self.std = np.sqrt(self.var + self.eps)
            self.xhat = (x - self.mu) / self.std
            self.rm = self.momentum * self.rm + (1 - self.momentum) * self.mu.reshape(-1)
            self.rv = self.momentum * self.rv + (1 - self.momentum) * self.var.reshape(-1)
        else:
            self.std = np.sqrt(self.rv.reshape(shp) + self.eps)
            self.xhat = (x - self.rm.reshape(shp)) / self.std
        return self.g.reshape(shp) * self.xhat + self.b.reshape(shp)

    def backward(self, d):
        """y = g*xhat + b；xhat = (x-mu)/std。
        dxh = d*g；dx = (dxh - E[dxh] - xhat*E[dxh*xhat]) / std（统计量按通道）"""
        axes = (0,) + tuple(range(2, self.x.ndim))
        shp = self._shape(self.x.ndim)
        self.dg[...] = (d * self.xhat).sum(axis=axes)
        self.db[...] = d.sum(axis=axes)
        g = self.g.reshape(shp)
        dxh = d * g
        return (dxh
                - dxh.mean(axis=axes, keepdims=True)
                - self.xhat * (dxh * self.xhat).mean(axis=axes, keepdims=True)
                ) / self.std


class ReLU:
    def pairs(self):
        return []

    def n_params(self):
        return 0

    def forward(self, x):
        self.m = x > 0
        return x * self.m

    def backward(self, d):
        return d * self.m


class MaxPool2:
    def pairs(self):
        return []

    def n_params(self):
        return 0

    def forward(self, x):
        N, C, H, W = x.shape
        self.shape = x.shape
        xr = x.reshape(N, C, H // 2, 2, W // 2, 2)
        out = xr.max(axis=(3, 5))
        self.mask = xr == out[:, :, :, None, :, None]
        self.cnt = self.mask.sum(axis=(3, 5), keepdims=True)
        return out

    def backward(self, d):
        N, C, H, W = self.shape
        dd = d[:, :, :, None, :, None] / self.cnt
        return (self.mask * dd).reshape(N, C, H, W)


class Flatten:
    def pairs(self):
        return []

    def n_params(self):
        return 0

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d):
        return d.reshape(self.shape)


class Linear:
    def __init__(self, nin, nout, rng, dtype=np.float32):
        std = np.sqrt(2.0 / nin)
        self.W = rng.normal(0, std, (nin, nout)).astype(dtype)
        self.b = np.zeros(nout, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def pairs(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW[...] = self.x.T @ d
        self.db[...] = d.sum(axis=0)
        return d @ self.W.T


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
    d = p
    d[np.arange(n), y] -= 1.0
    d /= n
    return loss, d, p


# ---------------------------------------------------------------- 网络
class CNN:
    """按 channels 逐段堆 Conv-[BN]-ReLU-Pool，再接两层全连接。"""

    def __init__(self, use_bn, channels=(16, 32, 64), fc_dim=64, seed=SEED):
        rng = np.random.default_rng(seed)
        self.use_bn = use_bn
        self.channels = tuple(channels)
        self.fc_dim = fc_dim
        c_in, h, w = 3, IMG, IMG
        ops, stages = [], []
        for c_out in self.channels:
            conv = Conv2(c_in, c_out, 3, 1, rng)
            bn = BatchNorm(c_out) if use_bn else None
            ops += [o for o in (conv, bn, ReLU(), MaxPool2()) if o is not None]
            h, w = h // 2, w // 2
            stages.append({"cin": c_in, "cout": c_out, "out_hw": (h, w),
                           "conv_params": conv.n_params(),
                           "bn_params": bn.n_params() if bn else 0,
                           "conv": conv, "bn": bn})
            c_in = c_out
        self.stages = stages
        self.first_conv = stages[0]["conv"]
        self.flat_dim = c_in * h * w
        self.fl = Flatten()
        self.f1 = Linear(self.flat_dim, fc_dim, rng)
        self.f2 = Linear(fc_dim, N_CLASS, rng)
        ops += [self.fl, self.f1, ReLU(), self.f2]
        self.ops = ops

    def mode(self, training):
        for o in self.ops:
            if isinstance(o, BatchNorm):
                o.training = training

    def forward(self, x):
        for o in self.ops:
            x = o.forward(x)
        return x

    def backward(self, d):
        for o in reversed(self.ops):
            d = o.backward(d)

    def n_params(self):
        return sum(o.n_params() for o in self.ops)

    def param_breakdown(self):
        conv = sum(s["conv_params"] for s in self.stages)
        bn = sum(s["bn_params"] for s in self.stages)
        fc = self.f1.n_params() + self.f2.n_params()
        return {"conv": conv, "bn": bn, "fc": fc, "total": conv + bn + fc}

    def arch_str(self):
        parts = []
        for s in self.stages:
            parts.append(f"Conv {s['cin']}->{s['cout']},3x3,pad1"
                         + ("+BN" if self.use_bn else "")
                         + f",pool2 -> {s['cout']}x{s['out_hw'][0]}x{s['out_hw'][1]}")
        parts.append(f"FC {self.flat_dim}->{self.fc_dim}")
        parts.append(f"FC {self.fc_dim}->{N_CLASS}")
        return " / ".join(parts)


class SGD:
    def __init__(self, model, lr, momentum=MOMENTUM):
        self.ps = []
        for o in model.ops:
            for p, g in o.pairs():
                self.ps.append((p, g))
        self.lr, self.momentum = lr, momentum
        self.v = [np.zeros_like(p) for p, _ in self.ps]

    def step(self):
        for i, (p, g) in enumerate(self.ps):
            self.v[i] = self.momentum * self.v[i] - self.lr * g
            p += self.v[i]

    def zero(self):
        for _, g in self.ps:
            g[...] = 0.0


def evaluate(model, X, y, bs=256):
    model.mode(False)
    n = X.shape[0]
    correct = 0
    preds = np.empty(n, dtype=np.int64)
    for s in range(0, n, bs):
        xb = X[s:s + bs]
        logits = model.forward(xb)
        preds[s:s + bs] = logits.argmax(axis=1)
    correct = int((preds == y).sum())
    model.mode(True)
    return correct / n, preds


def train_run(tag, Xtr, ytr, Xte, yte, use_bn, lr, epochs=EPOCHS, batch=BATCH,
              seed=SEED, channels=CHANNELS, note=""):
    print(f"\n===== 配置 {tag}：use_bn={use_bn}  lr={lr}  epochs={epochs} "
          f"channels={channels} {note} =====")
    model = CNN(use_bn=use_bn, channels=channels, seed=seed)
    pb = model.param_breakdown()
    print(f"[参数] 卷积 {pb['conv']:,} + BN {pb['bn']:,} + 全连接 {pb['fc']:,} "
          f"= 合计 {pb['total']:,}")
    opt = SGD(model, lr)
    n = Xtr.shape[0]
    hist = {"epoch": [], "train_loss": [], "test_acc": []}

    model.mode(False)
    acc0, _ = evaluate(model, Xte, yte)
    model.mode(True)
    print(f"[epoch 0] 未训练时测试准确率 = {acc0:.4f}")
    hist["epoch"].append(0)
    hist["train_loss"].append(float("nan"))
    hist["test_acc"].append(acc0)

    t_all = time.time()
    for ep in range(1, epochs + 1):
        idx = np.random.default_rng(seed + ep).permutation(n)
        tot, cnt = 0.0, 0
        t0 = time.time()
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb, yb = Xtr[b], ytr[b]
            logits = model.forward(xb)
            loss, d, _ = softmax_ce(logits, yb)
            opt.zero()
            model.backward(d)
            opt.step()
            tot += loss * xb.shape[0]
            cnt += xb.shape[0]
        acc, _ = evaluate(model, Xte, yte)
        hist["epoch"].append(ep)
        hist["train_loss"].append(tot / cnt)
        hist["test_acc"].append(acc)
        print(f"[{tag}] epoch {ep:2d}/{epochs}  loss={tot / cnt:.4f}  "
              f"test_acc={acc:.4f}  ({time.time() - t0:.1f}s)")

    acc_final, preds = evaluate(model, Xte, yte)
    cm = np.zeros((N_CLASS, N_CLASS), dtype=np.int64)
    for t, p in zip(yte, preds):
        cm[t, p] += 1
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print(f"[{tag}] 训练结束，最终测试准确率 = {acc_final:.4f}，总耗时 {time.time() - t_all:.1f}s")

    return {
        "tag": tag, "use_bn": use_bn, "lr": lr, "channels": list(model.channels),
        "arch": model.arch_str(),
        "stages": [{k: v for k, v in s.items() if k not in ("conv", "bn")}
                   for s in model.stages],
        "fc": {"flat_dim": model.flat_dim, "fc_dim": model.fc_dim,
               "f1_params": model.f1.n_params(), "f2_params": model.f2.n_params()},
        "params": pb, "hist": hist,
        "acc_final": float(acc_final),
        "acc_init": float(acc0),
        "confusion": cm, "per_class_acc": per_class, "preds": preds,
        "kernels": model.first_conv.W.copy(),
    }


def lr_scan(Xtr, ytr, Xte, yte, lrs=SCAN_LRS, epochs=SCAN_EPOCHS,
            n_train=SCAN_N_TRAIN, n_val=SCAN_N_VAL, seed=SEED):
    """阶段一：小规模子样本上扫学习率，回答"BN 到底把可用学习率推到哪儿"。"""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(ytr), min(n_train, len(ytr)), replace=False)
    Xs, ys = Xtr[sub], ytr[sub]
    vsub = rng.choice(len(yte), min(n_val, len(yte)), replace=False)
    Xv, yv = Xte[vsub], yte[vsub]
    print(f"\n########## 阶段一 学习率扫描 ##########")
    print(f"[scan] 训练子样本 {Xs.shape}  验证子样本 {Xv.shape}  "
          f"epochs={epochs}  batch={BATCH}")

    out = {"lrs": list(lrs), "epochs": epochs, "batch": BATCH,
           "n_train_sub": int(Xs.shape[0]), "n_val_sub": int(Xv.shape[0]),
           "momentum": MOMENTUM, "channels": list(CHANNELS),
           "bn": [], "nobn": [], "bn_hist": [], "nobn_hist": [],
           "bn_loss": [], "nobn_loss": []}

    for use_bn in (True, False):
        key = "bn" if use_bn else "nobn"
        for lr in lrs:
            t0 = time.time()
            model = CNN(use_bn=use_bn, channels=CHANNELS, seed=seed)
            opt = SGD(model, lr)
            hist, losses = [], []
            for ep in range(epochs):
                idx = np.random.default_rng(seed + ep).permutation(len(ys))
                tot, cnt = 0.0, 0
                for s in range(0, len(ys), BATCH):
                    b = idx[s:s + BATCH]
                    logits = model.forward(Xs[b])
                    loss, d, _ = softmax_ce(logits, ys[b])
                    opt.zero()
                    model.backward(d)
                    opt.step()
                    tot += loss * len(b); cnt += len(b)
                acc, _ = evaluate(model, Xv, yv)
                hist.append(round(acc, 4)); losses.append(round(tot / cnt, 4))
                print(f"[scan] BN={str(use_bn):5s} lr={lr:<6g} "
                      f"epoch {ep + 1}/{epochs}  loss={tot / cnt:.4f}  "
                      f"val_acc={acc * 100:5.1f}%  ({time.time() - t0:.0f}s)")
            out[key].append(hist[-1])
            out[f"{key}_hist"].append(hist)
            out[f"{key}_loss"].append(losses[-1])
    return out


def main():
    if not NPZ.exists():
        print("缺少 data/gtsrb32.npz，请先运行 code/prepare_data.py", file=sys.stderr)
        sys.exit(1)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = open(LOG, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, logf)
    t_start = time.time()

    d = np.load(NPZ)
    Xtr_u8, ytr = d["X_train"], d["y_train"]
    Xte_u8, yte = d["X_test"], d["y_test"]
    natural_counts = d["natural_counts"]
    print(f"[load] 训练 {Xtr_u8.shape}  测试 {Xte_u8.shape}  类别 {N_CLASS}")

    # 归一到 [0,1] 并转成 NCHW（卷积按 (N,C,H,W) 实现），再按通道用训练集统计量标准化
    Xtr = np.transpose(Xtr_u8.astype(np.float32) / 255.0, (0, 3, 1, 2))
    Xte = np.transpose(Xte_u8.astype(np.float32) / 255.0, (0, 3, 1, 2))
    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    print(f"[norm] 输入形状 {Xtr.shape}（N,C,H,W）")
    print(f"[norm] 通道均值 {mean.reshape(-1).round(4)}  "
          f"通道标准差 {std.reshape(-1).round(4)}")

    # ---------- 阶段一：学习率扫描 ----------
    scan = lr_scan(Xtr, ytr, Xte, yte)
    best_bn_i = int(np.argmax(scan["bn"]))
    best_nobn_i = int(np.argmax(scan["nobn"]))
    lr_bn = scan["lrs"][best_bn_i]
    lr_nobn = scan["lrs"][best_nobn_i]

    # ---------- 阶段二 ----------
    # 2x2 设计：{带 BN, 不带 BN} x {扫描最优的 0.01, 大 5 倍的 0.05}
    # A/B 之间只差 BN，用来隔离 BN 单独的贡献；C/D 把学习率放大 5 倍看容忍度
    print(f"\n########## 阶段二 全量训练集正式对照（2x2）##########")
    print(f"[scan 结论] 带 BN 最优 lr={lr_bn:g} -> {scan['bn'][best_bn_i] * 100:.1f}%")
    print(f"[scan 结论] 不带 BN 最优 lr={lr_nobn:g} -> {scan['nobn'][best_nobn_i] * 100:.1f}%")
    print(f"[scan 结论] 同一学习率 lr={lr_bn:g} 下：带 BN {scan['bn'][best_bn_i] * 100:.1f}%  "
          f"vs 不带 BN {scan['nobn'][best_bn_i] * 100:.1f}%")

    lr_big = LR_FINAL_BIG
    runs = [
        train_run("A_BN_lr0.01", Xtr, ytr, Xte, yte,
                  use_bn=True, lr=float(lr_bn),
                  note="[带 BN，扫描出的 BN 最优学习率]"),
        train_run("B_noBN_lr0.01", Xtr, ytr, Xte, yte,
                  use_bn=False, lr=float(lr_bn),
                  note="[不带 BN，学习率与 A 完全相同]"),
        train_run("C_BN_lr0.05", Xtr, ytr, Xte, yte,
                  use_bn=True, lr=float(lr_big),
                  note="[带 BN，学习率放大 5 倍]"),
        train_run("D_noBN_lr0.05", Xtr, ytr, Xte, yte,
                  use_bn=False, lr=float(lr_big),
                  note="[不带 BN，学习率放大 5 倍]"),
    ]

    # ---------- 参数账本 ----------
    # 一层 3x3 卷积(3->16) vs 要产出同样 16x32x32 个数的全连接层
    k, cin, cout = 3, 3, 16
    conv1 = (k * k * cin + 1) * cout                 # 448
    in_dims = cin * IMG * IMG                        # 3072
    rf_dims = k * k * cin                            # 27
    out_dims = cout * IMG * IMG                      # 16384
    fc_equiv = (in_dims + 1) * out_dims
    # 两种省法的乘积分解：局部连接省 local_ratio 倍，权重共享再省 share_ratio 倍
    local_ratio = (in_dims + 1) / (rf_dims + 1)
    share_ratio = out_dims / cout
    ledger = {
        "conv1_conv": f"{cin}->{cout}, {k}x{k}",
        "conv1_params": conv1,
        "conv1_weights": k * k * cin * cout,
        "conv1_bias": cout,
        "input_dims": in_dims,
        "receptive_field_dims": rf_dims,
        "output_dims": out_dims,
        "out_hw": [IMG, IMG],
        "fc_equiv_params": int(fc_equiv),
        "fc_equiv_ratio": float(fc_equiv / conv1),
        "local_ratio": float(local_ratio),
        "share_ratio": float(share_ratio),
        "params_per_output_channel_conv": rf_dims + 1,
        "params_per_output_channel_fc": in_dims + 1,
    }
    print(f"\n[账本] 一层 3x3 卷积({cin}->{cout})：{k}x{k}x{cin}={rf_dims} 个权重 "
          f"x {cout} 个输出通道 + {cout} 个偏置 = {conv1} 个参数")
    print(f"[账本] 它产出的输出是 {cout}x{IMG}x{IMG} = {out_dims:,} 个数")
    print(f"[账本] 等价全连接（{in_dims}->{out_dims:,}，每个输出都看全部 "
          f"{in_dims} 个输入）= {int(fc_equiv):,} 个参数")
    print(f"[账本] 倍数分解：局部连接 {in_dims + 1}/{rf_dims + 1} = {local_ratio:.2f} 倍"
          f" x 权重共享 {out_dims}/{cout} = {share_ratio:.0f} 倍"
          f" = {ledger['fc_equiv_ratio']:,.0f} 倍")

    # ---------- 汇总 ----------
    pb = runs[0]["params"]
    stats = {
        "dataset": {
            "name": "GTSRB (German Traffic Sign Recognition Benchmark)",
            "package": "GTSRB-Training_fixed.zip（IJCNN 2011 竞赛官方训练子集）",
            "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip",
            "md5": "513f3c79a4c5141765e10e952eaa2478",
            "size_bytes": 187490228,
            "n_class": N_CLASS,
            "img_size": IMG,
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "natural_counts": natural_counts.tolist(),
            "natural_min": int(natural_counts.min()),
            "natural_max": int(natural_counts.max()),
            "natural_total": int(natural_counts.sum()),
            "imbalance_ratio": float(natural_counts.max() / natural_counts.min()),
        },
        "net": {
            "channels": list(CHANNELS),
            "arch": runs[0]["arch"],
            "stages": runs[0]["stages"],
            "fc": runs[0]["fc"],
            "epochs": EPOCHS, "batch": BATCH, "momentum": MOMENTUM, "seed": SEED,
            "param_breakdown": pb,
            "n_params_bn": runs[0]["params"]["total"],
            "n_params_nobn": runs[1]["params"]["total"],
            "bn_extra_params": runs[0]["params"]["total"] - runs[1]["params"]["total"],
            "lr_final_big": LR_FINAL_BIG,
        },
        "ledger": ledger,
        "lr_scan": scan,
        "runs": [
            {
                "tag": r["tag"], "use_bn": r["use_bn"], "lr": r["lr"],
                "channels": r["channels"], "params": r["params"],
                "acc_init": r["acc_init"], "acc_final": r["acc_final"],
                "hist": r["hist"],
                "per_class_acc": r["per_class_acc"].tolist(),
            }
            for r in runs
        ],
        "class_names": CLASS_NAMES,
        "runtime_sec": None,
    }

    # 最容易混的两类（只看混淆矩阵非对角项）
    cmA = runs[0]["confusion"]
    off = cmA.copy()
    np.fill_diagonal(off, 0)
    flat = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:10]
    stats["top_confusions"] = [
        {"true": int(t), "pred": int(p), "count": int(cmA[t, p]),
         "true_name": CLASS_NAMES[t], "pred_name": CLASS_NAMES[p]}
        for t, p in flat if cmA[t, p] > 0
    ]
    print("[混淆] 最容易混的前 5 对：")
    for c in stats["top_confusions"][:5]:
        print(f"    {c['true_name']} -> 误判成 {c['pred_name']}  ({c['count']} 次)")

    stats["runtime_sec"] = round(time.time() - t_start, 1)
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {STATS}")

    # ---------- 画图素材 ----------
    arrays = {
        "confusion": runs[0]["confusion"],
        "per_class": runs[0]["per_class_acc"],
        "kernels": runs[0]["kernels"],
        "y_test": yte,
        "X_test": Xte_u8,
    }
    for r in runs:
        arrays[f"acc_{r['tag']}"] = np.array(r["hist"]["test_acc"])
        arrays[f"loss_{r['tag']}"] = np.array(r["hist"]["train_loss"])
        arrays[f"cm_{r['tag']}"] = r["confusion"]
    arrays["confusion_bn"] = runs[0]["confusion"]
    arrays["confusion_nobn"] = runs[1]["confusion"]
    arrays["per_class_bn"] = runs[0]["per_class_acc"]
    arrays["per_class_nobn"] = runs[1]["per_class_acc"]
    np.savez_compressed(PLOT, **arrays)
    print(f"[save] {PLOT}")
    print(f"[done] 总耗时 {stats['runtime_sec']}s")
    logf.close()


if __name__ == "__main__":
    main()
