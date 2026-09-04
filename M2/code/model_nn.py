# -*- coding: utf-8 -*-
"""
M2 神经网络：AI 生成的图，机器凭什么一眼认出来？

数据集：CIFAKE（真实照片 vs AI 生成图）
  - 60,000 张真实图（来自 CIFAR-10）+ 60,000 张 Stable Diffusion v1.4 生成图
  - 32×32 RGB，train 10 万 / test 2 万
  - 论文：Bird & Lotfi, "CIFAKE: Image Classification and Explainable
    Identification of AI-Generated Synthetic Images", IEEE Access, 2024.
    DOI 10.1109/ACCESS.2024.3356122
  - Kaggle 免 token 直链下载，约 105 MB

实验线（回答"多加的那一层到底图什么"）：
  A 原始像素 + 逻辑回归      一层线性模型，切不开
  B 原始像素 + 手写 MLP      加一层非线性，能学到像素组合
  C 卷积纹理 + 逻辑回归      只给 36 个参数的"看局部"结构，线性模型也能涨
  D 卷积纹理 + 手写 MLP      结构对了，再加非线性
  E 小样本 + 大网络          过拟合与三道防线

所有数字均为真实运行结果，落盘到 ../stats.json 与 ../figures/*.png
"""
import io
import json
import os
import re
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Okabe-Ito 色盲安全配色 ----
C = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "verm": "#D55E00", "purp": "#CC79A7",
}
plt.rcParams.update({
    "font.family": "PingFang SC",
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.size": 11,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

ZIP_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
           "birdy654/cifake-real-and-ai-generated-synthetic-images")
ZIP_PATH = os.path.join(DATA, "cifake.zip")

# 每类取多少张（控制实验规模，保证读者几分钟能跑完）
N_TR_CLS, N_TE_CLS = 6000, 1500
SEED = 42

# =====================================================================
# 1. 数据：自动下载 CIFAKE，直接从 zip 解出子集并缓存为 npy
# =====================================================================
CACHE = os.path.join(DATA, f"cifake_sub_{N_TR_CLS}_{N_TE_CLS}.npz")


def _name_key(name):
    """把 'train/REAL/0000 (10).jpg' 排成稳定顺序：先数字编号，再副本号。"""
    base = os.path.basename(name)
    m = re.match(r"^(\d+)(?:\s*\((\d+)\))?\.jpg$", base)
    if m:
        return (int(m.group(1)), int(m.group(2) or 0))
    return (10 ** 9, 0)


def load_subset():
    """返回 (Xtr, ytr, Xte, yte)，X 为 (N, 32, 32) 灰度 float32，0..1。
    标签 0=真实照片，1=AI 生成。"""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    if not os.path.exists(ZIP_PATH):
        print(f"== 下载 CIFAKE（约 105 MB，Kaggle 免 token 直链）==\n   {ZIP_URL}")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print(f"   已保存 {os.path.getsize(ZIP_PATH)/1e6:.1f} MB -> {ZIP_PATH}")

    z = zipfile.ZipFile(ZIP_PATH)
    names = z.namelist()

    def pick(split, cls, n):
        pre = f"{split}/{cls}/"
        cand = sorted([x for x in names
                       if x.startswith(pre) and x.lower().endswith(".jpg")],
                      key=_name_key)
        assert len(cand) >= n, f"{pre} 只有 {len(cand)} 张，取不到 {n} 张"
        return cand[:n]

    def read(paths):
        out = np.empty((len(paths), 32, 32), np.float32)
        for i, p in enumerate(paths):
            im = Image.open(io.BytesIO(z.read(p))).convert("RGB")
            a = np.asarray(im, dtype=np.float32) / 255.0
            out[i] = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        return out

    tr_r, tr_f = read(pick("train", "REAL", N_TR_CLS)), read(pick("train", "FAKE", N_TR_CLS))
    te_r, te_f = read(pick("test", "REAL", N_TE_CLS)), read(pick("test", "FAKE", N_TE_CLS))
    Xtr = np.vstack([tr_r, tr_f])
    Xte = np.vstack([te_r, te_f])
    ytr = np.array([0] * N_TR_CLS + [1] * N_TR_CLS)
    yte = np.array([0] * N_TE_CLS + [1] * N_TE_CLS)

    # 打乱（固定种子，保证可复现）
    rng = np.random.default_rng(SEED)
    p_tr, p_te = rng.permutation(len(Xtr)), rng.permutation(len(Xte))
    Xtr, ytr, Xte, yte = Xtr[p_tr], ytr[p_tr], Xte[p_te], yte[p_te]

    np.savez_compressed(CACHE, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


Xtr_img, ytr, Xte_img, yte = load_subset()
N_TR, N_TE = len(ytr), len(yte)
Xtr = Xtr_img.reshape(N_TR, -1)
Xte = Xte_img.reshape(N_TE, -1)
print(f"== 数据 ==  train {Xtr.shape}  test {Xte.shape}  "
      f"(0=真实照片 / 1=AI 生成)")


# =====================================================================
# 2. 手写两层 MLP：一次仿射 + ReLU + 一次仿射 + softmax
#    前向 = 逐层矩阵乘；反向 = 链式法则把梯度一层层乘回去
# =====================================================================
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP2:
    def __init__(self, din, dh, dout, lr=0.05, l2=0.0, seed=0):
        rng = np.random.default_rng(seed)
        # He 初始化：ReLU 网络的标准起手，保证信号不会被层层放大或压扁
        self.W1 = rng.standard_normal((din, dh)) * np.sqrt(2.0 / din)
        self.b1 = np.zeros(dh)
        self.W2 = rng.standard_normal((dh, dout)) * np.sqrt(2.0 / dh)
        self.b2 = np.zeros(dout)
        self.lr, self.l2 = lr, l2
        self._rng = np.random.default_rng(seed + 1)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)          # ReLU：没有它，整张网络塌回单层线性
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2, softmax(z2)

    def loss(self, a2, y):
        return float(-np.mean(np.log(a2[np.arange(len(y)), y] + 1e-12)))

    def backward(self, x, z1, a1, a2, y):
        m = x.shape[0]
        oh = np.zeros_like(a2)
        oh[np.arange(m), y] = 1
        dz2 = (a2 - oh) / m               # softmax 配交叉熵：梯度恰好是"预测 - 真值"
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T             # 链式法则：误差从输出层往隐藏层传
        dz1 = da1 * (z1 > 0)              # ReLU 的导数：正区间 1，否则 0
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    def step(self, x, y, dropout=0.0):
        z1, a1, z2, a2 = self.forward(x)
        if dropout > 0:
            mask = (self._rng.random(a1.shape) > dropout) / (1.0 - dropout)
            a1 = a1 * mask
            z2 = a1 @ self.W2 + self.b2
            a2 = softmax(z2)
        L = self.loss(a2, y)
        dW1, db1, dW2, db2 = self.backward(x, z1, a1, a2, y)
        if self.l2 > 0:                   # L2：权重越大，每次更新多扣一点
            dW1 += self.l2 * self.W1
            dW2 += self.l2 * self.W2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return L, float(np.linalg.norm(dW1)), float(np.linalg.norm(dW2))

    def train(self, X, y, Xv, yv, epochs=300, bs=0, dropout=0.0, log_every=25):
        """bs=0 表示全量梯度下降（损失与梯度范数都会单调下降，便于观察收敛）"""
        self.hist = {"loss": [], "tr": [], "va": [], "g1": [], "g2": []}
        rng = np.random.default_rng(SEED + 7)
        for e in range(epochs):
            if bs and bs < len(y):
                perm = rng.permutation(len(y))
                batches = [perm[s:s + bs] for s in range(0, len(perm), bs)]
            else:
                batches = [slice(None)]
            L = g1 = g2 = 0.0
            nb = 0
            for idx in batches:
                l, a, b = self.step(X[idx], y[idx], dropout)
                L += l; g1 += a; g2 += b; nb += 1
            tr = accuracy_score(y, self.predict(X))
            va = accuracy_score(yv, self.predict(Xv))
            self.hist["loss"].append(L / nb)
            self.hist["tr"].append(tr)
            self.hist["va"].append(va)
            self.hist["g1"].append(g1 / nb)
            self.hist["g2"].append(g2 / nb)
            if log_every and (e % log_every == 0 or e == epochs - 1):
                print(f"  epoch {e:3d} | loss {L/nb:.4f} | train {tr:.4f} | "
                      f"test {va:.4f} | ||dW1|| {g1/nb:.4f} ||dW2|| {g2/nb:.4f}")
        return self.hist

    def predict(self, X):
        return self.forward(X)[3].argmax(1)

    def acc(self, X, y):
        return accuracy_score(y, self.predict(X))


# =====================================================================
# 3. 卷积：4 个固定的 3×3 核，权值共享
# =====================================================================
def make_filters():
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32) / 4.0
    laplace = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32) / 4.0
    blob = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32) / 8.0
    return np.stack([sobel_x, sobel_x.T, laplace, blob])   # (4, 3, 3)


FILT = make_filters()
N_FILT = FILT.shape[0]


def conv_relu_pool(imgs, filt):
    """imgs (N,32,32) -> (N, F*15*15)。同一个 3×3 核在 30×30 个位置上复用。"""
    N = imgs.shape[0]
    F = filt.shape[0]
    out = np.empty((N, F, 30, 30), np.float32)
    for f in range(F):
        k = filt[f]
        for i in range(30):
            for j in range(30):
                out[:, f, i, j] = (imgs[:, i:i + 3, j:j + 3] * k).sum(axis=(1, 2))
    out = np.maximum(out, 0.0)                                  # ReLU
    out = out.reshape(N, F, 15, 2, 15, 2).mean(axis=(3, 5))     # 2×2 平均池化
    return out.reshape(N, -1)


# ---------------------------------------------------------------------
# 可训练的卷积网络：卷积 -> ReLU -> 2×2 平均池化 -> 全连接 -> softmax
# 卷积核不再由人指定，而是跟全连接层一起被反向传播教出来
# ---------------------------------------------------------------------
def im2col(imgs, k=3):
    """imgs (B,32,32) -> (B*30*30, 9)，把每个 3×3 小块摊成一行（支持批量）"""
    B = imgs.shape[0]
    H = W = 32 - k + 1
    s0, s1, s2 = imgs.strides
    patches = np.lib.stride_tricks.as_strided(
        imgs, shape=(B, H, W, k, k), strides=(s0, s1, s2, s1, s2))
    return patches.reshape(B * H * W, k * k)


class ConvNet:
    """卷积核 F (nF,9) 与全连接 W (nF*15*15, 2) 都由梯度下降学出来。"""

    def __init__(self, nF=8, lr_f=0.05, lr_w=0.05, l2=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.nF = nF
        self.F = rng.standard_normal((nF, 9)) * 0.1   # 随机起手，不是人设计的
        self.W = rng.standard_normal((nF * 15 * 15, 2)) * np.sqrt(2.0 / (nF * 15 * 15))
        self.b = np.zeros(2)
        self.lr_f, self.lr_w, self.l2 = lr_f, lr_w, l2

    def forward(self, imgs):
        B = imgs.shape[0]
        pat = im2col(imgs)                                    # (B*900, 9)
        z1 = (pat @ self.F.T).reshape(B, 900, self.nF)        # (B, 30*30, nF)
        z1 = z1.transpose(0, 2, 1).reshape(B, self.nF, 30, 30)
        a1 = np.maximum(0.0, z1)                              # ReLU
        pool = a1.reshape(B, self.nF, 15, 2, 15, 2).mean(axis=(3, 5))
        flat = pool.reshape(B, -1)
        z2 = flat @ self.W + self.b
        return pat, z1, a1, flat, z2, softmax(z2)

    def loss(self, p, y):
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    def step(self, imgs, y):
        B = len(y)
        pat, z1, a1, flat, z2, p = self.forward(imgs)
        L = self.loss(p, y)
        oh = np.zeros_like(p)
        oh[np.arange(B), y] = 1
        dz2 = (p - oh) / B                                    # 输出层梯度
        dW = flat.T @ dz2
        db = dz2.sum(axis=0)
        dflat = dz2 @ self.W.T                                # 往回传
        dp = dflat.reshape(B, self.nF, 15, 15) / 4.0          # 反平均池化
        dz1 = np.repeat(np.repeat(dp, 2, axis=2), 2, axis=3) * (z1 > 0)  # 反 ReLU
        dz1f = dz1.reshape(B, self.nF, 900).transpose(0, 2, 1).reshape(B * 900, self.nF)
        dF = dz1f.T @ pat                                     # 卷积核的梯度
        if self.l2 > 0:
            dF += self.l2 * self.F
            dW += self.l2 * self.W
        self.F -= self.lr_f * dF
        self.W -= self.lr_w * dW
        self.b -= self.lr_w * db
        return L, float(np.linalg.norm(dF)), float(np.linalg.norm(dW))

    def train(self, X, y, Xv, yv, epochs=30, bs=512, log_every=5):
        self.hist = {"loss": [], "tr": [], "va": [], "gF": []}
        rng = np.random.default_rng(SEED + 13)
        for e in range(epochs):
            perm = rng.permutation(len(y))
            L = gF = 0.0
            nb = 0
            for s in range(0, len(perm), bs):
                idx = perm[s:s + bs]
                l, g, _ = self.step(X[idx], y[idx])
                L += l; gF += g; nb += 1
            tr = accuracy_score(y, self.predict(X))
            va = accuracy_score(yv, self.predict(Xv))
            self.hist["loss"].append(L / nb)
            self.hist["tr"].append(tr)
            self.hist["va"].append(va)
            self.hist["gF"].append(gF / nb)
            if log_every and (e % log_every == 0 or e == epochs - 1):
                print(f"  epoch {e:3d} | loss {L/nb:.4f} | train {tr:.4f} | "
                      f"test {va:.4f} | ||dF|| {gF/nb:.4f}")
        return self.hist

    def predict(self, X):
        return self.forward(X)[5].argmax(1)

    def acc(self, X, y):
        return accuracy_score(y, self.predict(X))


# =====================================================================
# 4. 实验 A/B：原始像素
# =====================================================================
print("\n== A. 原始像素 + 逻辑回归（一层线性模型）==")
lin = LogisticRegression(max_iter=2000)
lin.fit(Xtr, ytr)
acc_lin = accuracy_score(yte, lin.predict(Xte))
print(f"  测试准确率 {acc_lin:.4f}")

print("\n== B. 原始像素 + 手写 MLP（1024 -> 128 -> 2，mini-batch）==")
mlp_raw = MLP2(1024, 128, 2, lr=0.05, seed=SEED)
h_raw = mlp_raw.train(Xtr, ytr, Xte, yte, epochs=200, bs=256, log_every=50)
acc_mlp_raw = mlp_raw.acc(Xte, yte)
loss_first, loss_last = h_raw["loss"][0], h_raw["loss"][-1]
g1_first, g1_last = h_raw["g1"][0], h_raw["g1"][-1]

# =====================================================================
# 5. 实验 C/D：卷积纹理特征
# =====================================================================
print("\n== C. 人设计的卷积核（Sobel/Laplace，固定不练）+ 逻辑回归 ==")
Ftr = conv_relu_pool(Xtr_img, FILT)
Fte = conv_relu_pool(Xte_img, FILT)
lin_c = LogisticRegression(max_iter=2000)
lin_c.fit(Ftr, ytr)
acc_conv_lin = accuracy_score(yte, lin_c.predict(Fte))
print(f"  特征维度 {Ftr.shape[1]} | 测试准确率 {acc_conv_lin:.4f}")

print("\n== D. 卷积核交给反向传播自己学（8 个 3x3，端到端）==")
net = ConvNet(nF=8, lr_f=0.1, lr_w=0.1, seed=SEED)
h_net = net.train(Xtr_img, ytr, Xte_img, yte, epochs=200, bs=256, log_every=20)
acc_net = net.acc(Xte_img, yte)
net_loss_first, net_loss_last = h_net["loss"][0], h_net["loss"][-1]
gF_first, gF_last = h_net["gF"][0], h_net["gF"][-1]
learned_F = net.F.copy()

# 权值共享的参数账
conv_params = 8 * 3 * 3
fc_equiv = 32 * 32 * (8 * 30 * 30)      # 全连接铺满同样扫描范围
share_ratio = fc_equiv / conv_params

# =====================================================================
# 6. sklearn 参照（不参与论证，只说明手写实现没跑偏）
# =====================================================================
print("\n== 参照：sklearn MLP（同样原始像素）==")
skm = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=SEED)
skm.fit(Xtr, ytr)
acc_skm = accuracy_score(yte, skm.predict(Xte))
print(f"  测试准确率 {acc_skm:.4f}")

# =====================================================================
# 7. 实验 E：过拟合（小样本 + 大网络）与三道防线
# =====================================================================
N_SMALL = 400
print(f"\n== E. 过拟合：只给 {N_SMALL} 张图，隐藏层 256（26 万个参数 vs 400 个样本）==")
rs = np.random.default_rng(SEED)
idx = rs.choice(N_TR, N_SMALL, replace=False)
Xs, ys = Xtr[idx], ytr[idx]
N_PARAM = 1024 * 256 + 256 * 2

net_none = MLP2(1024, 256, 2, lr=0.05, seed=SEED)
h_none = net_none.train(Xs, ys, Xte, yte, epochs=400, bs=256, log_every=200)
net_l2 = MLP2(1024, 256, 2, lr=0.05, l2=3e-3, seed=SEED)
h_l2 = net_l2.train(Xs, ys, Xte, yte, epochs=400, bs=256, log_every=400)
net_drop = MLP2(1024, 256, 2, lr=0.05, seed=SEED)
h_drop = net_drop.train(Xs, ys, Xte, yte, epochs=400, bs=256, dropout=0.5, log_every=400)

es_epoch = int(np.argmax(h_none["va"])) + 1
es_val = h_none["va"][es_epoch - 1]
print(f"  无防线  末轮 train {h_none['tr'][-1]:.4f} test {h_none['va'][-1]:.4f} | "
      f"早停点 epoch {es_epoch} test {es_val:.4f}")
print(f"  L2      末轮 train {h_l2['tr'][-1]:.4f} test {h_l2['va'][-1]:.4f}")
print(f"  Dropout 末轮 train {h_drop['tr'][-1]:.4f} test {h_drop['va'][-1]:.4f}")

# =====================================================================
# 8. 配图
# =====================================================================
# fig1：你能分出来吗（真实 vs AI 生成，各 8 张并排）
fig, axes = plt.subplots(2, 8, figsize=(9.6, 2.9))
for r, (lab, title) in enumerate([(0, "真实照片"), (1, "AI 生成")]):
    src = Xtr_img[(ytr == r)]
    for c in range(8):
        ax = axes[r, c]
        ax.imshow(src[c], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(title, fontsize=10)
fig.suptitle("上面是真实照片，下面是 AI 生成的。32×32，你能看出区别吗？", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_samples.png"))
plt.close()

# fig2：网络结构
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
inp_x = 0.08
ax.imshow(Xtr_img[0], cmap="gray", extent=(inp_x, inp_x + 0.13, 0.32, 0.55), vmin=0, vmax=1)
ax.add_patch(plt.Rectangle((inp_x, 0.32), 0.13, 0.23, fc="none", ec=C["black"], lw=0.8))
ax.text(inp_x + 0.065, 0.62, "输入 32×32=1024", ha="center", fontsize=10)
hid_x = 0.50
hid_ys = np.linspace(0.16, 0.62, 12)
for yy in hid_ys:
    ax.plot(hid_x, yy, "o", ms=8, color=C["orange"], mec=C["black"], mew=0.3, zorder=3)
ax.text(hid_x, 0.70, "隐藏层 128", ha="center", fontsize=10)
out_x = 0.86
out_ys = [0.30, 0.48]
for yy, txt, col in zip(out_ys, ["真实", "AI"], [C["blue"], C["verm"]]):
    ax.plot(out_x, yy, "o", ms=13, color=col, mec=C["black"], mew=0.4, zorder=3)
    ax.text(out_x + 0.045, yy, txt, va="center", fontsize=10)
ax.text(out_x, 0.70, "输出 2", ha="center", fontsize=10)
for yh in hid_ys[::3]:
    ax.plot([inp_x + 0.13, hid_x], [0.435, yh], "-", color=C["black"], lw=0.4, alpha=0.4)
    for yo in out_ys:
        ax.plot([hid_x, out_x], [yh, yo], "-", color=C["black"], lw=0.25, alpha=0.28)
ax.text((inp_x + 0.13 + hid_x) / 2, 0.92, "W₁ (1024×128)", ha="center", fontsize=10, color=C["verm"])
ax.text((hid_x + out_x) / 2, 0.92, "W₂ (128×2)", ha="center", fontsize=10, color=C["verm"])
ax.text(0.5, 0.06,
        "前向：x →(W₁) z₁ →(ReLU) a₁ →(W₂) z₂ →(softmax) 概率\n"
        "一层 = 一次仿射变换 + 一次非线性；少了非线性，再深也只是单层",
        ha="center", fontsize=9.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_arch.png"))
plt.close()

# fig3：激活函数与梯度
x = np.linspace(-4, 4, 200)
sig = 1 / (1 + np.exp(-x)); sigd = sig * (1 - sig)
tanh = np.tanh(x); tanhd = 1 - tanh ** 2
relu = np.maximum(0, x); relud = (x > 0).astype(float)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
axes[0].plot(x, sig, color=C["blue"], label="sigmoid")
axes[0].plot(x, tanh, color=C["orange"], label="tanh")
axes[0].plot(x, relu, color=C["green"], label="ReLU")
axes[0].axhline(0, color="grey", lw=0.5); axes[0].axvline(0, color="grey", lw=0.5)
axes[0].set_title("激活函数（非线性的来源）"); axes[0].legend(fontsize=9)
axes[1].plot(x, sigd, color=C["blue"], label="sigmoid′")
axes[1].plot(x, tanhd, color=C["orange"], label="tanh′")
axes[1].plot(x, relud, color=C["green"], label="ReLU′")
axes[1].axhline(0, color="grey", lw=0.5); axes[1].axvline(0, color="grey", lw=0.5)
axes[1].set_title("它们的导数（反向传播的弹药）"); axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_activations.png"))
plt.close()

# fig4：四种打法的准确率对比 + 学出来的卷积核长什么样
fig = plt.figure(figsize=(11.2, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
ax = fig.add_subplot(gs[0, 0])
names = ["原始像素\n+ 逻辑回归", "原始像素\n+ 手写 MLP",
         "人设计的卷积核\n+ 逻辑回归", "学出来的卷积核\n（端到端）"]
vals = [acc_lin, acc_mlp_raw, acc_conv_lin, acc_net]
cols = [C["verm"], C["orange"], C["sky"], C["green"]]
b = ax.bar(names, vals, color=cols, width=0.62)
ax.axhline(0.5, color="grey", ls=":", lw=1)
ax.text(3.40, 0.508, "瞎猜 0.5", fontsize=8.5, color="grey")
ax.set_ylim(0.45, 0.92)
ax.set_ylabel("测试准确率")
ax.set_title("同一批图，四种打法", fontsize=11)
for rect, v in zip(b, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f"{v:.3f}",
            ha="center", fontsize=9.5)

axk = fig.add_subplot(gs[0, 1])
n_show = 8
grid = np.zeros((1 + n_show * 2, 1 + n_show * 2))
for i in range(n_show):
    r0, c0 = 1 + (i // 4) * 2, 1 + (i % 4) * 2
    f = learned_F[i].reshape(3, 3)
    f = f / (np.abs(f).max() + 1e-9)
    grid[r0:r0 + 3 - 1, c0:c0 + 3 - 1] = 0  # 占位
    for a in range(3):
        for bb in range(3):
            grid[r0 + a, c0 + bb] = f[a, bb]
axk.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1)
axk.set_title("反向传播学出来的 8 个 3×3 核", fontsize=11)
axk.set_xticks([]); axk.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_compare.png"))
plt.close()

# fig5：过拟合曲线
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ep = np.arange(1, len(h_none["tr"]) + 1)
ax.plot(ep, h_none["tr"], color=C["blue"], label="训练准确率（无防线）")
ax.plot(ep, h_none["va"], color=C["orange"], ls="--", label="测试准确率（无防线）")
ax.plot(ep, h_drop["va"], color=C["green"], ls="-.", label="测试准确率（Dropout）")
ax.axvline(es_epoch, color=C["verm"], ls=":", lw=1.4,
           label=f"早停点（epoch {es_epoch}）")
ax.set_xlabel("训练轮数 epoch"); ax.set_ylabel("准确率")
ax.set_title(f"只给 {N_SMALL} 张图、{N_PARAM//1000} 万个参数：训练集背到满分，测试集上不去",
             fontsize=10.5)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_overfit.png"))
plt.close()

# fig6：ConvNet 训练动态（损失降下去、准确率涨上来 = 机器在学自己的卷积核）
fig, ax = plt.subplots(figsize=(7.4, 4.0))
epn = np.arange(1, len(h_net["loss"]) + 1)
ax.plot(epn, h_net["loss"], color=C["verm"], lw=1.8, label="训练损失")
ax.set_xlabel("训练轮数 epoch"); ax.set_ylabel("交叉熵损失", color=C["verm"])
ax.tick_params(axis="y", labelcolor=C["verm"])
ax2 = ax.twinx()
ax2.plot(epn, h_net["va"], color=C["green"], lw=1.8, ls="--", label="测试准确率")
ax2.plot(epn, h_net["tr"], color=C["blue"], lw=1.2, alpha=0.6, label="训练准确率")
ax2.set_ylabel("准确率"); ax2.set_ylim(0.4, 1.0)
ax.set_title("ConvNet 自己学卷积核：损失降下去，准确率涨上来", fontsize=10.5)
ax2.legend(fontsize=8, loc="center right")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_convnet_curve.png"))
plt.close()

# =====================================================================
# 9. stats.json
# =====================================================================
stats = {
    "dataset": "CIFAKE (real vs AI-generated), Kaggle birdy654",
    "paper": "Bird & Lotfi, IEEE Access 2024, DOI 10.1109/ACCESS.2024.3356122",
    "full_size": {"train": 100000, "test": 20000, "img": "32x32 RGB"},
    "subset_used": {
        "n_train": int(N_TR), "n_test": int(N_TE),
        "per_class_train": N_TR_CLS, "per_class_test": N_TE_CLS, "channels": "grayscale",
    },
    # A / B：直接在原始像素上做
    "raw_pixel": {
        "logistic_regression": round(float(acc_lin), 4),
        "manual_mlp": round(float(acc_mlp_raw), 4),
        "sklearn_mlp": round(float(acc_skm), 4),
    },
    # C：人设计的 4 个固定核（Sobel / Laplace / blob），不训练，只给线性模型看局部
    "human_designed_kernel": {
        "n_filters": int(N_FILT),
        "conv_params": int(N_FILT * 3 * 3),
        "feature_dim": int(Ftr.shape[1]),
        "logistic_regression": round(float(acc_conv_lin), 4),
    },
    # D：卷积核不再由人指定，跟全连接层一起被反向传播教出来
    "learned_conv_net": {
        "n_filters": int(net.nF),
        "conv_params": int(net.nF * 3 * 3),
        "fc_equiv_params": int(fc_equiv),
        "weight_sharing_ratio": int(share_ratio),
        "test_acc": round(float(acc_net), 4),
        "loss_first": round(float(net_loss_first), 4),
        "loss_last": round(float(net_loss_last), 4),
        "grad_F_first": round(float(gF_first), 4),
        "grad_F_last": round(float(gF_last), 4),
    },
    # B 的细节：mini-batch SGD，损失持续下降（证明在学）
    "manual_mlp_detail": {
        "arch": "1024->128->2", "lr": 0.05, "epochs": 200, "batch_size": 256,
        "loss_first": round(float(loss_first), 4),
        "loss_last": round(float(loss_last), 4),
        "grad_W1_first": round(float(g1_first), 4),
        "grad_W1_last": round(float(g1_last), 4),
    },
    "overfit_demo": {
        "n_small": N_SMALL, "hidden": 256, "epochs": 400, "n_param": int(N_PARAM),
        "no_def_train": round(float(h_none["tr"][-1]), 4),
        "no_def_test": round(float(h_none["va"][-1]), 4),
        "earlystop_epoch": es_epoch,
        "earlystop_test": round(float(es_val), 4),
        "l2_train": round(float(h_l2["tr"][-1]), 4),
        "l2_test": round(float(h_l2["va"][-1]), 4),
        "dropout_train": round(float(h_drop["tr"][-1]), 4),
        "dropout_test": round(float(h_drop["va"][-1]), 4),
    },
}
with open(os.path.join(ROOT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("\n== stats.json 已写出 ==")
print(json.dumps(stats, ensure_ascii=False, indent=2))
