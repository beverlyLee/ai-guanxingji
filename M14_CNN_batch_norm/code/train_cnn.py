"""
M14 从零 numpy CNN 实战：卷积层构造 + BatchNormalization + 参数对比 + 网络测试效果
数据集：Olivetti 真实人脸（sklearn 直连下载，400 张 64x64，40 人）
二分类任务：前 20 人 (label 0) vs 后 20 人 (label 1)
目的：端到端实跑，产出文章每个数字的可复现来源（坑2 / 坑5）。
所有数字同时出现在：脚本打印 / stats.json / 配图标注（三处一致）。
"""

import json
import time
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------------
# 基础算子：卷积(im2col) / 最大池化 / 批归一化 / 全连接 / softmax 交叉熵
# ----------------------------------------------------------------------------
def im2col(x, kh, kw, stride=1, pad=0):
    N, C, H, W = x.shape
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    cols = np.zeros((N, C, kh, kw, Ho, Wo))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = xp[:, :, i:i + Ho * stride:stride, j:j + Wo * stride:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, C * kh * kw)
    return cols, N, Ho, Wo


def col2im(cols, x_shape, kh, kw, stride=1, pad=0):
    N, C, H, W = x_shape
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    cols = cols.reshape(N, Ho, Wo, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + Ho * stride:stride, j:j + Wo * stride:stride] += cols[:, :, i, j, :, :]
    return xp[:, :, pad:H + pad, pad:W + pad] if pad else xp


class Conv2d:
    def __init__(self, cin, cout, k=3, stride=1, pad=1, lr=0.01):
        self.k, self.stride, self.pad = k, stride, pad
        self.W = RNG.standard_normal((cout, cin, k, k)) * np.sqrt(2.0 / (cin * k * k))
        self.b = np.zeros(cout)
        self.lr = lr

    def forward(self, x):
        self.x = x
        cols, N, Ho, Wo = im2col(x, self.k, self.k, self.stride, self.pad)
        self.cols = cols
        Wc = self.W.reshape(self.W.shape[0], -1)
        out = cols @ Wc.T
        out = out.reshape(N, Ho, Wo, self.W.shape[0]).transpose(0, 3, 1, 2)
        out += self.b.reshape(1, -1, 1, 1)
        return out

    def backward(self, dout):
        F = self.W.shape[0]
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        dWc = dout_r.T @ self.cols
        self.dW = dWc.reshape(self.W.shape)
        self.db = dout_r.sum(axis=0)
        dx_cols = dout_r @ self.W.reshape(F, -1)
        return col2im(dx_cols, self.x.shape, self.k, self.k, self.stride, self.pad)

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class MaxPool:
    def __init__(self, size=2, stride=2):
        self.size, self.stride = size, stride

    def forward(self, x):
        N, C, H, W = x.shape
        Ho, Wo = H // self.stride, W // self.stride
        out = np.zeros((N, C, Ho, Wo))
        self.mask = np.zeros_like(x)
        for i in range(Ho):
            for j in range(Wo):
                win = x[:, :, i * self.stride:i * self.stride + self.size, j * self.stride:j * self.stride + self.size]
                m = win.max(axis=(2, 3))
                out[:, :, i, j] = m
                for c in range(C):
                    flat = win[:, c].reshape(N, self.size * self.size)
                    idx = flat.argmax(axis=1)
                    for n in range(N):
                        ii, jj = divmod(int(idx[n]), self.size)
                        self.mask[n, c, i * self.stride + ii, j * self.stride + jj] = 1.0
        self.x = x
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        Ho, Wo = dout.shape[2], dout.shape[3]
        dx = np.zeros_like(self.x)
        for i in range(Ho):
            for j in range(Wo):
                d = dout[:, :, i, j]
                for c in range(C):
                    dx[:, c, i * self.stride:i * self.stride + self.size, j * self.stride:j * self.stride + self.size] += d[:, c][:, None, None]
        return dx * self.mask


class BatchNorm:
    def __init__(self, C, eps=1e-5):
        self.C, self.eps = C, eps
        self.gamma = np.ones(C)
        self.beta = np.zeros(C)

    def forward(self, x, training=True):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        xr = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mu = xr.mean(0)
        var = xr.var(0)
        self.std = np.sqrt(var + self.eps)
        self.xhat = (xr - mu) / self.std
        self.xr = xr
        out = self.xhat * self.gamma + self.beta
        return out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        dr = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        M = dr.shape[0]
        self.dgamma = (dr * self.xhat).sum(0)
        self.dbeta = dr.sum(0)
        dxhat = dr * self.gamma
        dx = (1.0 / (M * self.std)) * (M * dxhat - dxhat.sum(0) - self.xhat * (dxhat * self.xhat).sum(0))
        return dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


class FC:
    def __init__(self, din, dout):
        self.W = RNG.standard_normal((din, dout)) * np.sqrt(2.0 / din)
        self.b = np.zeros(dout)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(0)
        return dout @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


def softmax(x):
    e = x - x.max(1, keepdims=True)
    e = np.exp(e)
    return e / e.sum(1, keepdims=True)


def cross_entropy(pred, y):
    return -np.mean(np.log(pred + 1e-12) * y)


def onehot(y, k):
    o = np.zeros((y.size, k))
    o[np.arange(y.size), y.astype(int)] = 1
    return o


# ----------------------------------------------------------------------------
# 网络
# ----------------------------------------------------------------------------
class Net:
    def __init__(self, use_bn=True, lr=0.01):
        self.use_bn = use_bn
        self.lr = lr
        self.conv1 = Conv2d(1, 8, k=3, stride=1, pad=1)
        self.pool1 = MaxPool(2, 2)
        self.bn1 = BatchNorm(8) if use_bn else None
        self.conv2 = Conv2d(8, 16, k=3, stride=1, pad=1)
        self.pool2 = MaxPool(2, 2)
        self.bn2 = BatchNorm(16) if use_bn else None
        self.fc = FC(16 * 16 * 16, 2)

    def forward(self, x, training=True):
        self.x0 = x
        self.z1 = self.conv1.forward(x)
        self.r1 = np.maximum(0, self.z1)
        self.p1 = self.pool1.forward(self.r1)
        if self.use_bn:
            self.b1 = self.bn1.forward(self.p1, training)
        else:
            self.b1 = self.p1
        self.z2 = self.conv2.forward(self.b1)
        self.r2 = np.maximum(0, self.z2)
        self.p2 = self.pool2.forward(self.r2)
        if self.use_bn:
            self.b2 = self.bn2.forward(self.p2, training)
        else:
            self.b2 = self.p2
        self.flat = self.b2.reshape(self.b2.shape[0], -1)
        self.z3 = self.fc.forward(self.flat)
        self.prob = softmax(self.z3)
        return self.prob

    def backward(self, dout):
        dfc = self.fc.backward(dout)
        d = dfc.reshape(self.b2.shape)
        if self.use_bn:
            d = self.bn2.backward(d)
        d = self.pool2.backward(d)
        d = d * (self.z2 > 0)
        d = self.conv2.backward(d)
        if self.use_bn:
            d = self.bn1.backward(d)
        d = self.pool1.backward(d)
        d = d * (self.z1 > 0)
        self.conv1.backward(d)

    def step(self, lr):
        self.conv1.step(lr)
        self.conv2.step(lr)
        self.fc.step(lr)
        if self.use_bn:
            self.bn1.step(lr)
            self.bn2.step(lr)

    def count_params(self):
        p = 0
        p += self.conv1.W.size + self.conv1.b.size
        p += self.conv2.W.size + self.conv2.b.size
        p += self.fc.W.size + self.fc.b.size
        return int(p)

    def conv_only_params(self):
        # 第一层卷积如果用全连接替代（64x64 输入 -> 8 通道）的参数量对比
        conv_w = self.conv1.W.size + self.conv1.b.size
        fc_w = (64 * 64) * 8 + 8
        return conv_w, fc_w


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def load_data():
    faces = fetch_olivetti_faces(data_home="/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/ai-guanxingji/M14_CNN_batch_norm/data")
    X = faces.images.astype(np.float32) / 255.0          # (400,64,64)
    y = faces.target                                       # 0..39
    idx0 = np.where(y < 20)[0]
    idx1 = np.where(y >= 20)[0]
    X = np.concatenate([X[idx0], X[idx1]])[:, None, :, :]  # (400,1,64,64)
    y = np.concatenate([np.zeros(len(idx0)), np.ones(len(idx1))]).astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    return Xtr, Xte, ytr, yte


def train(net, Xtr, ytr, epochs=40, batch=20, lr=0.01, verbose=True):
    Ytr = onehot(ytr, 2)
    loss_hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = RNG.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            prob = net.forward(xb, training=True)
            loss = cross_entropy(prob, yb)
            dout = (prob - yb) / len(xb)
            net.backward(dout)
            net.step(lr)
        prob = net.forward(Xtr, training=False)
        loss_hist.append(float(cross_entropy(prob, Ytr)))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            acc = (prob.argmax(1) == ytr).mean()
            print(f"  epoch {ep+1:2d}/{epochs}  train_loss={loss_hist[-1]:.4f}  train_acc={acc:.3f}")
    return loss_hist, time.time() - t0


def evaluate(net, Xte, yte):
    prob = net.forward(Xte, training=False)
    pred = prob.argmax(1)
    acc = (pred == yte).mean()
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(yte, pred):
        cm[int(t), int(p)] += 1
    return float(acc), cm


def main():
    Xtr, Xte, ytr, yte = load_data()
    print(f"data: train={len(Xtr)} test={len(Xte)}  (前20人 vs 后20人, 各200张)")
    print(f"input shape: {Xtr.shape[1:]}")

    # 参数量对比（第一层卷积 vs 等价的全连接）
    demo = Net(use_bn=True)
    conv_w, fc_w = demo.conv_only_params()
    total = demo.count_params()
    conv_ratio = fc_w / conv_w

    print(f"\n[参数量对比] 第一层若用全连接(64x64->8)需 {fc_w} 个权重(+8偏置)")
    print(f"[参数量对比] 第一层用卷积(3x3, 权值共享)仅需 {conv_w} 个权重(+8偏置)")
    print(f"[参数量对比] 卷积相对全连接减少 {conv_ratio:.1f} 倍参数")
    print(f"[参数量对比] 整个网络总参数: {total}")

    stats = {
        "dataset": "Olivetti Faces (sklearn fetch_olivetti_faces)",
        "task": "二分类: 前20人 vs 后20人",
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "input_shape": [1, 64, 64],
        "conv1_params": int(conv_w),
        "equiv_fc1_params": int(fc_w),
        "conv_param_reduction_x": round(float(conv_ratio), 1),
        "total_params": int(total),
        "arch": "Conv(3x3,8)+ReLU+Pool2+[BN]+Conv(3x3,16)+ReLU+Pool2+[BN]+FC(4096->2)",
    }

    # 两组训练：带 BN / 不带 BN，对比收敛
    print("\n=== 训练 A: 带 BatchNorm (lr=0.01) ===")
    net_bn = Net(use_bn=True, lr=0.01)
    loss_bn, t_bn = train(net_bn, Xtr, ytr, epochs=40, lr=0.01)
    acc_bn, cm_bn = evaluate(net_bn, Xte, yte)
    print(f"  test_acc={acc_bn:.4f}  time={t_bn:.1f}s  final_loss={loss_bn[-1]:.4f}")

    print("\n=== 训练 B: 不带 BatchNorm (lr=0.005, 收敛更慢) ===")
    net_nb = Net(use_bn=False, lr=0.005)
    loss_nb, t_nb = train(net_nb, Xtr, ytr, epochs=40, lr=0.005)
    acc_nb, cm_nb = evaluate(net_nb, Xte, yte)
    print(f"  test_acc={acc_nb:.4f}  time={t_nb:.1f}s  final_loss={loss_nb[-1]:.4f}")

    # 不带 BN 用更大 lr 会怎样（验证 BN 允许更大学习率）
    print("\n=== 训练 C: 不带 BN 但 lr=0.01 (对照 exploded) ===")
    net_nb2 = Net(use_bn=False, lr=0.01)
    try:
        loss_nb2, _ = train(net_nb2, Xtr, ytr, epochs=40, lr=0.01, verbose=False)
        acc_nb2, cm_nb2 = evaluate(net_nb2, Xte, yte)
        exploded = bool(np.isnan(loss_nb2[-1]) or loss_nb2[-1] > 5)
    except Exception as e:
        acc_nb2, cm_nb2, loss_nb2, exploded = 0.0, np.zeros((2, 2), int), [float("nan")], True
        print(f"  lr=0.01 无BN 训练异常: {e}")

    stats.update({
        "acc_with_bn": round(acc_bn, 4),
        "acc_without_bn": round(acc_nb, 4),
        "acc_without_bn_biglr": round(acc_nb2, 4),
        "without_bn_biglr_exploded": bool(exploded),
        "loss_bn": [round(x, 4) for x in loss_bn],
        "loss_without_bn": [round(x, 4) for x in loss_nb],
        "cm_with_bn": cm_bn.tolist(),
        "cm_without_bn": cm_nb.tolist(),
        "time_with_bn_s": round(t_bn, 1),
        "time_without_bn_s": round(t_nb, 1),
    })

    # 特征图：用测试集第 0 张图看第一层 8 个卷积核的输出
    sample = Xte[:1]
    prob = net_bn.forward(sample, training=False)
    conv1_out = net_bn.conv1.forward(sample)  # (1,8,64,64)
    filters = net_bn.conv1.W.copy()           # (8,1,3,3)

    np.savez(
        "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/ai-guanxingji/M14_CNN_batch_norm/data/plot_data.npz",
        loss_bn=np.array(loss_bn),
        loss_nb=np.array(loss_nb),
        conv1_out=conv1_out[0],        # (8,64,64)
        filters=filters[:, 0],          # (8,3,3)
        sample_img=sample[0, 0],        # (64,64)
        cm_bn=cm_bn,
        cm_nb=cm_nb,
    )
    with open("/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/ai-guanxingji/M14_CNN_batch_norm/data/stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== 汇总 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n已保存: stats.json, plot_data.npz")


if __name__ == "__main__":
    main()
