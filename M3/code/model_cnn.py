# -*- coding: utf-8 -*-
"""
M3 卷积神经网络：AI 看皮肤病越来越准，可它盯着的可能不是那颗痣

数据集：HAM10000（皮肤镜图像，7 类皮肤病）
  - 10,015 张皮肤镜 RGB 图像，600×450
  - 7 类（dx 字段）：nv 6705 / mel 1113 / bkl 1099 / bcc 514 /
    akiec 327 / vasc 142 / df 115
  - 来源：维也纳医科大学皮肤科 + 澳大利亚 Queensland，历时 20 年
  - 论文：Tschandl et al., "The HAM10000 dataset...", Scientific Data, 2018.
    DOI 10.1038/s41597-018-0180-z（PMID 30106392）
  - Kaggle 免 token 直链下载，约 2.5 GB
  - 注意：该数据集以浅色皮肤人群为主（论文明确记载），与热点中的
    肤色偏见直接呼应

实验线（回答"卷积网络的视野到底有多大"）：
  EXP-0 七类样本（fig1）
  EXP-1 手写卷积前向：单核滑动 + 特征图可视化（fig2）
  EXP-2 感受野计算：单层/堆叠/加池化的视野扩张（fig3）
  EXP-3 核大小×步长×padding：输出尺寸公式验证 + 边缘覆盖次数（fig4）
  EXP-4 池化前后：尺寸减半、感受野翻倍、平移不变性（fig5）
  EXP-5 VGG 思想：小核堆叠 vs 大核，参数量与精度（fig6）
  EXP-6 残差连接：深层普通块 vs 残差块，训练曲线（fig7）
  EXP-7 网络看哪里：类激活高亮 + 背景调暗翻转实验（fig8）

所有数字均为真实运行结果，落盘到 ../stats.json 与 ../figures/*.png
"""
import json
import os

import numpy as np
from PIL import Image

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

CLASS_NAMES = {
    "nv": "色素痣 nv", "mel": "黑色素瘤 mel", "bkl": "良性角化病 bkl",
    "bcc": "基底细胞癌 bcc", "akiec": "光化性角化病 akiec",
    "vasc": "血管病变 vasc", "df": "皮肤纤维瘤 df",
}
STATS = {}


# ===================== 数据加载 =====================
def load_data():
    """加载 extract_from_parts.py 产出的 npz 与 manifest。

    npz: ids (2113,)  y (2113,)（mel=1 / nv=0）  X (2113, 96, 96) uint8
    前一半是 mel（1113 张全取），后一半是 nv（default_rng(42) 抽样 1000 张）。
    """
    with open(os.path.join(DATA, "extracted_manifest.json")) as f:
        manifest = json.load(f)
    z = np.load(os.path.join(DATA, "ham_gray96.npz"))
    return z["ids"], z["y"], z["X"], manifest


def downscale32(X96):
    """96×96 → 32×32 双线性缩放（训练用）。"""
    from PIL import Image
    out = np.empty((len(X96), 32, 32), dtype=np.float32)
    for i in range(len(X96)):
        img = Image.fromarray(X96[i])
        out[i] = np.asarray(img.resize((32, 32), Image.BILINEAR),
                            dtype=np.float32) / 255.0
    return out


# ===================== 卷积核心（手写，im2col 加速）=====================
def im2col(X, kh, kw, stride, pad):
    """X: (N,C,H,W) → cols: (N*oh*ow, C*kh*kw)。"""
    N, C, H, W = X.shape
    oh = (H - kh + 2 * pad) // stride + 1
    ow = (W - kw + 2 * pad) // stride + 1
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else X
    cols = np.empty((N, oh, ow, C, kh, kw), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, :, i, j] = Xp[
                :, :, i:i + oh * stride:stride,
                j:j + ow * stride:stride].transpose(0, 2, 3, 1)
    return cols.reshape(N * oh * ow, C * kh * kw), (N, C, oh, ow)


def conv2d(X, K, stride=1, pad=0):
    """X: (N,C,H,W)，K: (Co,C,kh,kw) → out (N,Co,oh,ow)。

    返回 (out, cols)，cols 供反向传播复用。
    """
    Co, Ci, kh, kw = K.shape
    cols, (N, C, oh, ow) = im2col(X, kh, kw, stride, pad)
    out = cols @ K.reshape(Co, -1).T          # (N*oh*ow, Co)
    out = out.reshape(N, oh, ow, Co).transpose(0, 3, 1, 2)
    return out, cols


def conv_backward(dout, X, K, stride, pad, cols):
    """dout: (N,Co,oh,ow) → dX (N,C,H,W), dK (Co,C,kh,kw)。"""
    N, C, H, W = X.shape
    Co, Ci, kh, kw = K.shape
    oh, ow = dout.shape[2], dout.shape[3]
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, Co)   # (N*oh*ow, Co)
    dK = (dout_flat.T @ cols).reshape(K.shape)
    dcols = (dout_flat @ K.reshape(Co, -1)).reshape(
        N, oh, ow, C, kh, kw)
    Hp = H + 2 * pad; Wp = W + 2 * pad
    dXp = np.zeros((N, C, Hp, Wp), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            dXp[:, :, i:i + oh * stride:stride,
                j:j + ow * stride:stride] += dcols[:, :, :, :, i, j].transpose(
                    0, 3, 1, 2)
    if pad:
        dXp = dXp[:, :, pad:-pad, pad:-pad]
    return dXp, dK


def maxpool2x2(X):
    """X: (N,C,H,W) → (N,C,H/2,W/2)，2×2 池化步长 2。"""
    N, C, H, W = X.shape
    return X.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def maxpool2x2_backward(dout, X):
    """最大池化反向：梯度只流向 2×2 窗口里的最大值位置。"""
    N, C, H, W = X.shape
    H2, W2 = H // 2, W // 2
    Xf = X.reshape(N, C, H2, 2, W2, 2).reshape(N, C, H2, W2, 4)
    am = Xf.argmax(axis=4)                                  # (N,C,H2,W2)
    dx = np.zeros_like(Xf)
    n_idx = np.arange(N)[:, None, None, None]
    c_idx = np.arange(C)[None, :, None, None]
    h_idx = np.arange(H2)[None, None, :, None]
    w_idx = np.arange(W2)[None, None, None, :]
    dx[n_idx, c_idx, h_idx, w_idx, am] = dout
    return dx.reshape(N, C, H, W)


def relu(X):
    return np.maximum(X, 0.0)


# ---- 数值梯度校验（小尺寸，保证手写反向传播正确）----
def grad_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 1, 6, 6))
    K = rng.normal(size=(3, 1, 3, 3)) * 0.1
    out, cols = conv2d(X, K, stride=2, pad=1)
    dout = rng.normal(size=out.shape)
    dX, dK = conv_backward(dout, X, K, stride=2, pad=1, cols=cols)
    eps = 1e-6
    def loss(K_):
        o, _ = conv2d(X, K_, stride=2, pad=1)
        return float((o * dout).sum())
    num = np.zeros_like(K)
    for a in range(K.shape[0]):
        for b in range(K.shape[2]):
            for c in range(K.shape[3]):
                Kp = K.copy(); Kp[a, 0, b, c] += eps
                Km = K.copy(); Km[a, 0, b, c] -= eps
                num[a, 0, b, c] = (loss(Kp) - loss(Km)) / (2 * eps)
    err_k = np.abs(num - dK).max() / (np.abs(num).max() + 1e-12)
    assert err_k < 1e-6, f"核梯度校验失败: {err_k}"
    # 输入梯度也校验
    def loss_x(X_):
        o, _ = conv2d(X_, K, stride=2, pad=1)
        return float((o * dout).sum())
    numx = np.zeros_like(X)
    for a in range(6):
        for b in range(6):
            Xp = X.copy(); Xp[0, 0, a, b] += eps
            Xm = X.copy(); Xm[0, 0, a, b] -= eps
            numx[0, 0, a, b] = (loss_x(Xp) - loss_x(Xm)) / (2 * eps)
    err_x = np.abs(numx[0, 0] - dX[0, 0]).max() / (
        np.abs(numx[0, 0]).max() + 1e-12)
    assert err_x < 1e-6, f"输入梯度校验失败: {err_x}"
    return float(max(err_k, err_x))


# ===================== 感受野 / 输出尺寸（纯计算）=====================
def rf_trace(layers):
    """逐层累计感受野与跳跃步长。layers: [(type, k, s, pad), ...]。
    type: 'conv' / 'pool'（pool 固定 k=2, s=2, pad=0）。"""
    rf, jump = 1, 1
    trace = []
    for layer in layers:
        typ, k, s, pad = layer[0], layer[1], layer[2], layer[3]
        kk = 2 if typ == "pool" else k
        ss = 2 if typ == "pool" else s
        rf = rf + (kk - 1) * jump
        jump = jump * ss
        trace.append((typ, rf, jump))
    return trace, jump


def out_size(W, k, s, p):
    return (W - k + 2 * p) // s + 1


def edge_coverage(k, W=32, pad=0):
    """卷积下，输入平面每个像素被核窗口盖住的次数（中心 vs 角）。"""
    cnt = np.zeros((W, W), dtype=np.int64)
    oh = out_size(W, k, 1, pad)
    for i in range(oh):
        for j in range(oh):
            r0 = i - pad; c0 = j - pad
            r0c, c0c = max(r0, 0), max(c0, 0)
            r1c, c1c = min(r0 + k, W), min(c0 + k, W)
            cnt[r0c:r1c, c0c:c1c] += 1
    return cnt


# ===================== numpy 小 CNN =====================
def he_init(n, shape, rng):
    return rng.normal(0, np.sqrt(2.0 / n), size=shape).astype(np.float32)


class SmallCNN:
    """可配置卷积网络（输入单通道灰度 (N,1,S,S)）。

    blocks 元素：
      ("conv", k, s, pad, ch)  卷积 + ReLU
      ("pool",)                2×2 最大池化
      ("res", k, pad, ch)      残差块：relu(conv(relu(conv(x))) + x)
    末端固定：flatten → FC(hidden) → ReLU → FC(1) → sigmoid
    """
    def __init__(self, blocks, fc_hidden=32, seed=0):
        self.blocks = blocks
        self.rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        ci, cur_size = 1, 32
        for b in blocks:
            if b[0] == "conv":
                _, k, s, pad, ch = b
                self.W.append(he_init(ci * k * k, (ch, ci, k, k), self.rng))
                self.b.append(np.zeros(ch, dtype=np.float32))
                ci = ch
                cur_size = out_size(cur_size, k, s, pad)
            elif b[0] == "res":
                _, k, pad, ch = b
                for _ in range(2):
                    self.W.append(he_init(ci * k * k, (ch, ci, k, k),
                                          self.rng))
                    self.b.append(np.zeros(ch, dtype=np.float32))
                    ci = ch
                cur_size = out_size(cur_size, k, 1, pad)   # 同尺寸卷积×2
            elif b[0] == "pool":
                cur_size = cur_size // 2
        self.fc1_in = ci * cur_size * cur_size
        self.W.append(he_init(self.fc1_in, (fc_hidden, self.fc1_in),
                              self.rng))
        self.b.append(np.zeros(fc_hidden, dtype=np.float32))
        self.W.append(he_init(fc_hidden, (1, fc_hidden), self.rng))
        self.b.append(np.zeros(1, dtype=np.float32))
        self.n_conv_w = len(self.W) - 2   # 卷积/残差层的权重个数

    def forward(self, X, cache=False):
        self._c = {"convs": [], "masks": [], "pools": [], "res": []}
        h = X
        wi = 0
        for b in self.blocks:
            if b[0] == "conv":
                _, k, s, pad, ch = b
                z, cols = conv2d(h, self.W[wi], s, pad)
                a = relu(z)
                if cache:
                    self._c["convs"].append((h, wi, s, pad, cols))
                    self._c["masks"].append(z > 0)
                h = a
                wi += 1
            elif b[0] == "res":
                _, k, pad, ch = b
                x_in = h
                z1, cols1 = conv2d(h, self.W[wi], 1, pad)
                a1 = relu(z1)
                m1 = z1 > 0
                wi += 1
                z2, cols2 = conv2d(a1, self.W[wi], 1, pad)
                wi += 1
                h = relu(z2 + x_in)
                if cache:
                    self._c["res"].append((x_in, wi - 2, z1, m1, z2,
                                           cols1, cols2))
            elif b[0] == "pool":
                if cache:
                    self._c["pools"].append(h)
                h = maxpool2x2(h)
        self._h_last = h
        flat = h.reshape(len(X), -1)
        z1 = flat @ self.W[-2].T + self.b[-2]
        a1 = relu(z1)
        z2 = a1 @ self.W[-1].T + self.b[-1]
        p = 1.0 / (1.0 + np.exp(-z2[:, 0]))
        if cache:
            self._c.update({"flat_h": h, "fcz1": z1, "fcz2": z2, "p": p})
        return p

    def backward(self, y, lr):
        """BCE 损失 + SGD。返回 batch 平均损失。"""
        N = len(y)
        c = self._c
        p = c["p"] if "p" in c else None
        dz2 = (p - y).astype(np.float32).reshape(N, 1) / N
        dW_last = dz2.T @ relu(c["fcz1"])
        db_last = dz2.sum(axis=0)
        da1 = dz2 @ self.W[-1]
        dz1 = da1 * (c["fcz1"] > 0)
        dW_fc = dz1.T @ c["flat_h"].reshape(N, -1)
        db_fc = dz1.sum(axis=0)
        dflat = dz1 @ self.W[-2]
        dh = dflat.reshape(c["flat_h"].shape)

        grads_w = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        grads_w[-1], grads_b[-1] = dW_last, db_last
        grads_w[-2], grads_b[-2] = dW_fc, db_fc

        for b in reversed(self.blocks):
            if b[0] == "pool":
                Xin = c["pools"].pop()
                dh = maxpool2x2_backward(dh, Xin)
            elif b[0] == "conv":
                x_in, widx, s, pad, cols = c["convs"].pop()
                mask = c["masks"].pop()
                dz = dh * mask
                dX, dK = conv_backward(dz, x_in, self.W[widx], s, pad, cols)
                grads_w[widx] = dK
                grads_b[widx] = dz.sum(axis=(0, 2, 3))
                dh = dX
            elif b[0] == "res":
                x_in, widx, z1, m1, z2, cols1, cols2 = c["res"].pop()
                dz2r = dh * (z2 + x_in > 0)          # 相加后 ReLU 的掩码
                da1, dK2 = conv_backward(dz2r, x_in * 0 + relu(z1),
                                         self.W[widx + 1], 1, b[2], cols2)
                grads_w[widx + 1] = dK2
                grads_b[widx + 1] = dz2r.sum(axis=(0, 2, 3))
                dz1 = da1 * m1
                dx1, dK1 = conv_backward(dz1, x_in, self.W[widx], 1, b[2],
                                         cols1)
                grads_w[widx] = dK1
                grads_b[widx] = dz1.sum(axis=(0, 2, 3))
                dh = dx1 + dz2r     # 残差直通车道：梯度原样加回输入
        for i in range(len(self.W)):
            self.W[i] -= lr * grads_w[i]
            self.b[i] -= lr * grads_b[i]
        loss = -np.mean(y * np.log(p + 1e-12)
                        + (1 - y) * np.log(1 - p + 1e-12))
        return float(loss)

    def conv_params(self):
        """卷积/残差层参数量（不含全连接）。"""
        n = sum(int(w.size + b.size)
                for w, b in zip(self.W[:self.n_conv_w],
                                self.b[:self.n_conv_w]))
        return n


def train_model(model, X, y, Xva, yva, epochs=6, bs=64, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    tr_losses, va_accs = [], []
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        losses = []
        for s in range(0, len(X), bs):
            bidx = idx[s:s + bs]
            model.forward(X[bidx], cache=True)
            losses.append(model.backward(y[bidx], lr))
        tr_losses.append(float(np.mean(losses)))
        pva = model.forward(Xva)
        va_accs.append(float(((pva > 0.5) == yva).mean()))
    return tr_losses, va_accs


def eval_acc(model, X, y):
    p = model.forward(X)
    return float(((p > 0.5) == y).mean()), p


# ===================== 实 验 =====================
def exp0_samples(meta):
    """fig1：七类样本。"""
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 7, figsize=(16.5, 2.8))
    counts = {}
    for ax, (dx, name) in zip(axes, CLASS_NAMES.items()):
        ids = [r["image_id"] for r in meta if r["dx"] == dx]
        counts[dx] = len(ids)
        iid = ids[rng.integers(len(ids))]
        img = Image.open(find_image(iid)).resize((224, 224))
        ax.imshow(img)
        ax.set_title(f"{name}\n{len(ids)} 张")
        ax.axis("off")
    fig.suptitle("HAM10000 的 7 类皮肤病样本（各类张数标注在标题）", y=1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_samples.png"), bbox_inches="tight")
    plt.close(fig)
    STATS["dataset"] = {"total": len(meta), "counts": counts}


def exp0_samples(manifest):
    """fig1：七类样本（用 showcase/ 下预存的 224×224 RGB 图）。"""
    import glob
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 7, figsize=(16.5, 2.8))
    counts = manifest["counts"]
    for ax, (dx, name) in zip(axes, CLASS_NAMES.items()):
        files = sorted(glob.glob(os.path.join(DATA, "showcase",
                                              f"{dx}_*.jpg")))
        ax.imshow(Image.open(files[int(rng.integers(len(files)))]))
        ax.set_title(f"{name}\n{counts[dx]} 张")
        ax.axis("off")
    fig.suptitle("HAM10000 的 7 类皮肤病样本（各类张数标注在标题）", y=1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_samples.png"), bbox_inches="tight")
    plt.close(fig)
    STATS["dataset"] = {"total": sum(counts.values()), "counts": counts}


def exp1_conv_featuremap(ids, y, X96):
    """fig2：手写卷积前向。一张 mel 图上跑 4 种核，画出特征图。"""
    mel_id = str(ids[y == 1][0])
    img = X96[y == 1][0] / 255.0
    rng = np.random.default_rng(5)
    kernels = {
        "水平边缘核": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=np.float32),
        "垂直边缘核": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=np.float32),
        "斑点核(拉普拉斯)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=np.float32),
        "随机初始化核\n(训练前)": rng.normal(0, 0.5, (3, 3)).astype(np.float32),
    }
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("输入：黑色素瘤皮肤镜图\n(96×96 灰度)")
    for j, (name, K) in enumerate(kernels.items()):
        out, _ = conv2d(img[None, None], K[None, None])
        axes[j + 1].imshow(out[0, 0], cmap="magma")
        axes[j + 1].set_title(f"{name}\n→ 特征图 94×94")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("EXP-1 手写卷积：一个 3×3 核（9 个参数）扫过全图，"
                 "滑出一整张特征图。前三个核抓边缘和斑点，随机核暂时只是乱纹，"
                 "要靠训练把它变成有用的核", y=1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_conv_featuremap.png"),
                bbox_inches="tight")
    plt.close(fig)
    STATS["exp1"] = {
        "image_id": mel_id,
        "shared_params": 9,
        "fc_equiv_params_96px": 96 * 96 * 9,
    }
    return kernels


def exp2_receptive_field():
    """fig3：感受野逐层扩张可视化。"""
    configs = [
        ("单层 3×3", [("conv", 3, 1, 0, 8)]),
        ("两层 3×3", [("conv", 3, 1, 0, 8), ("conv", 3, 1, 0, 8)]),
        ("三层 3×3", [("conv", 3, 1, 0, 8)] * 3),
        ("两层3×3+池化", [("conv", 3, 1, 0, 8), ("conv", 3, 1, 0, 8),
                          ("pool", 2, 2, 0, 0)]),
        ("两层3×3+池化\n+一层3×3", [("conv", 3, 1, 0, 8),
                                    ("conv", 3, 1, 0, 8),
                                    ("pool", 2, 2, 0, 0),
                                    ("conv", 3, 1, 0, 8)]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    G = 11
    rf_values = []
    for ax, (name, layers) in zip(axes, configs):
        trace, _ = rf_trace(layers)
        rf = trace[-1][1]
        rf_values.append(rf)
        for i in range(G + 1):
            ax.plot([0, G], [i, i], color="#BBBBBB", lw=0.6, zorder=1)
            ax.plot([i, i], [0, G], color="#BBBBBB", lw=0.6, zorder=1)
        c = (G - rf) / 2
        ax.add_patch(plt.Rectangle((c, c), rf, rf, facecolor=C["verm"],
                                   alpha=0.75, zorder=2))
        ax.add_patch(plt.Rectangle((c, c), rf, rf, fill=False,
                                   edgecolor=C["black"], lw=1.4, zorder=3))
        ax.set_title(f"{name}\n感受野 = {rf}×{rf}", fontsize=11)
        ax.set_xlim(-0.3, G + 0.3); ax.set_ylim(G + 0.3, -0.3)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("EXP-2 感受野：越深/加池化，一个输出神经元看到的输入区域越大。"
                 "它只由核大小、步长、池化和层数决定，与输入图多大无关", y=1.14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_receptive_field.png"),
                bbox_inches="tight")
    plt.close(fig)
    STATS["exp2"] = {
        "configs": [n.replace("\n", "") for n, _ in configs],
        "rf_values": rf_values,
    }


def exp3_kernel_stride_padding():
    """fig4：输出尺寸公式验证 + 边缘覆盖次数。"""
    checks = []
    for (W, k, s, p) in [(224, 3, 1, 0), (224, 3, 1, 1), (224, 5, 2, 2),
                         (224, 7, 2, 3), (96, 3, 2, 1), (32, 5, 1, 2)]:
        assert out_size(W, k, s, p) == (W - k + 2 * p) // s + 1
        checks.append({"W": W, "k": k, "s": s, "p": p,
                       "out": out_size(W, k, s, p)})
    cnt_valid = edge_coverage(3, W=32, pad=0)
    cnt_pad1 = edge_coverage(3, W=32, pad=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    im0 = axes[0].imshow(cnt_valid, cmap="magma")
    axes[0].set_title("3×3 无填充：每个像素\n被核盖住几次（32×32）")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].axis("off")
    im1 = axes[1].imshow(cnt_pad1, cmap="magma")
    axes[1].set_title("3×3 填充 1 圈：角落从 1 次升到 4 次\n"
                      f"（中心仍是 {cnt_pad1[16, 16]} 次）")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].axis("off")
    for k, col in zip([3, 5, 7], [C["blue"], C["orange"], C["verm"]]):
        ss = [1, 2, 4]
        outs = [out_size(224, k, s, 0) for s in ss]
        axes[2].plot(range(3), outs, "o-", color=col, label=f"{k}×{k}核")
        for x, o in zip(range(3), outs):
            axes[2].annotate(str(o), (x, o), textcoords="offset points",
                             xytext=(0, 6), fontsize=9)
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(["步长1", "步长2", "步长4"])
    axes[2].set_ylabel("特征图边长（输入 224）")
    axes[2].set_title("核越大 / 步长越大\n特征图缩得越快（都无填充）")
    axes[2].legend(frameon=False)
    fig.suptitle("EXP-3 输出尺寸 out = (W − K + 2P)/S + 1（脚本逐项 assert 验证），"
                 "以及无填充时边缘像素被看到的次数远少于中心", y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_kernel_stride_padding.png"),
                bbox_inches="tight")
    plt.close(fig)
    STATS["exp3"] = {
        "checks": checks,
        "corner_cover_valid": int(cnt_valid[0, 0]),
        "center_cover_valid": int(cnt_valid[16, 16]),
        "corner_cover_pad1": int(cnt_pad1[0, 0]),
    }


def exp4_pooling(ids, y, X96, kernel):
    """fig5：池化前后：尺寸、计算量，以及位置信息的丢失。"""
    mel_id = str(ids[y == 1][1])
    img = X96[y == 1][1] / 255.0
    f1, _ = conv2d(img[None, None], kernel[None, None])
    feat = relu(f1)[0, 0]                              # 94×94
    pool = maxpool2x2(feat[None, None])[0, 0]          # 47×47
    H2 = feat.shape[0] // 2
    # 位置丢失演示：取全图响应最强的真实池化窗口，窗口内 4 个值任意排列，
    # 池化输出都是同一个最大值（max 只认值、不认位置）
    fr = feat.reshape(H2, 2, H2, 2).reshape(-1, 4)
    w0 = fr[int(np.argmax(fr.max(axis=1)))].copy()
    perms = [w0, w0[[1, 0, 3, 2]], w0[[3, 2, 1, 0]], w0[[2, 3, 0, 1]]]
    maxes = [float(p.max()) for p in perms]
    assert max(maxes) == min(maxes), "max 不因窗口内排列而变，演示失败"
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8),
                             gridspec_kw={"width_ratios": [1, 1.15, 1, 1.25]})
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("输入 96×96")
    axes[1].imshow(feat, cmap="magma")
    axes[1].set_title(f"卷积特征图 {feat.shape[0]}×{feat.shape[0]}")
    axes[2].imshow(pool, cmap="magma")
    axes[2].set_title(f"MaxPool 2×2 后 {pool.shape[0]}×{pool.shape[0]}\n"
                      "尺寸减半，下层计算量省约 4 倍")
    labels = ["原窗口", "左右交换", "对角交换", "旋转排列"]
    axes[3].bar(labels, [float(v) for v in w0],
                color=[C["blue"]] * 4, width=0.55)
    axes[3].axhline(maxes[0], color=C["verm"], lw=1.2, ls="--")
    axes[3].text(0.02, maxes[0], f"  输出都是 {maxes[0]:.3f}",
                 color=C["verm"], fontsize=10, va="bottom")
    axes[3].set_title(f"响应最强的真实窗口（值 {w0[0]:.2f}/{w0[1]:.2f}/"
                      f"{w0[2]:.2f}/{w0[3]:.2f}）\n"
                      "窗口内怎么排，池化输出都一样 → 位置信息丢了")
    axes[3].tick_params(axis="x", labelsize=9)
    for ax in axes[:3]:
        ax.axis("off")
    fig.suptitle("EXP-4 池化的收益与代价：更小、更省、后续感受野扩张更快，"
                 "但只记最大值、不记位置", y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_pooling.png"), bbox_inches="tight")
    plt.close(fig)
    STATS["exp4"] = {
        "image_id": mel_id,
        "size_before": int(feat.shape[0]),
        "size_after": int(pool.shape[0]),
        "next_layer_compute_ratio": round(
            (feat.shape[0] / pool.shape[0]) ** 2, 1),
        "strongest_window_values": [float(v) for v in w0],
        "window_perm_max": maxes[0],
    }


def exp5_vgg(Xtr, ytr, Xva, yva, Xte, yte):
    """fig6：小核堆叠 vs 大核（VGG 思想）：参数量与精度。"""
    archs = {
        "一层 3×3（浅）": [("conv", 3, 1, 1, 8), ("pool",),
                            ("conv", 3, 1, 1, 16), ("pool",)],
        "两层 3×3（VGG）": [("conv", 3, 1, 1, 8), ("conv", 3, 1, 1, 8),
                            ("pool",), ("conv", 3, 1, 1, 16),
                            ("conv", 3, 1, 1, 16), ("pool",)],
        "一层 5×5": [("conv", 5, 1, 2, 8), ("pool",),
                     ("conv", 5, 1, 2, 16), ("pool",)],
    }
    results = {}
    trained = {}
    for name, blocks in archs.items():
        m = SmallCNN(blocks, seed=0)
        n_conv = m.conv_params()
        tl, va = train_model(m, Xtr, ytr, Xva, yva, epochs=6, lr=0.05)
        acc, _ = eval_acc(m, Xte, yte)
        results[name] = {"conv_params": n_conv, "test_acc": acc,
                         "val_acc_final": va[-1]}
        trained[name] = m
    rf_2x3 = rf_trace([("conv", 3, 1, 0, 8)] * 2)[0][-1][1]
    rf_5 = rf_trace([("conv", 5, 1, 0, 8)])[0][-1][1]
    assert rf_2x3 == rf_5 == 5
    p233 = 2 * 9 * 8 * 8       # 同通道数 8：两个 3×3 = 18C²
    p55 = 25 * 8 * 8           # 一个 5×5 = 25C²
    p33 = 9 * 8 * 8            # 一个 3×3 = 9C²（视野只有 3）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    stage_names = ["一个 3×3\n(视野 3)", "两个 3×3 堆叠\n(视野 5)",
                   "一个 5×5\n(视野 5)"]
    stage_params = [p33, p233, p55]
    stage_colors = [C["sky"], C["orange"], C["blue"]]
    bars = axes[0].bar(stage_names, stage_params, color=stage_colors,
                       width=0.55)
    axes[0].set_ylabel("一级卷积核参数量（8 进 8 出）")
    axes[0].set_title("同样的视野，堆小核更省参数")
    for b, v in zip(bars, stage_params):
        axes[0].text(b.get_x() + 0.25, v, f"{v:,}", fontsize=10)
    names = list(results.keys())
    accs = [results[n]["test_acc"] for n in names]
    colors = [C["sky"], C["orange"], C["blue"]]
    axes[1].bar(names, accs, color=colors, width=0.5)
    axes[1].set_ylabel("测试集准确率")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_title("mel vs nv 测试集准确率（同样训练 6 轮）")
    for i, v in enumerate(accs):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    axes[1].tick_params(axis="x", labelsize=9)
    fig.suptitle("EXP-5 VGG 的道理：两个 3×3 的感受野 = 一个 5×5"
                 f"（同通道数下 {p233} vs {p55} 参数），还多一次非线性", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_vgg_vs_bigkernel.png"),
                bbox_inches="tight")
    plt.close(fig)
    STATS["exp5"] = {
        "results": results,
        "rf_two_3x3": int(rf_2x3), "rf_one_5x5": int(rf_5),
        "params_1x3x3_same8ch": p33,
        "params_2x3x3_same8ch": p233, "params_1x5x5_same8ch": p55,
    }
    return trained


def exp6_residual(Xtr, ytr, Xva, yva, Xte, yte):
    """fig7：深层普通块 vs 残差块，训练可训性对照。"""
    n_conv = 6
    plain_blocks = []
    res_blocks = []
    for _ in range(n_conv // 2):
        plain_blocks += [("conv", 3, 1, 1, 8), ("conv", 3, 1, 1, 8),
                         ("pool",)]
        res_blocks += [("res", 3, 1, 8), ("pool",)]
    m_plain = SmallCNN(plain_blocks, seed=0)
    m_res = SmallCNN(res_blocks, seed=0)
    tl_p, va_p = train_model(m_plain, Xtr, ytr, Xva, yva, epochs=6, lr=0.05)
    tl_r, va_r = train_model(m_res, Xtr, ytr, Xva, yva, epochs=6, lr=0.05)
    acc_p, _ = eval_acc(m_plain, Xte, yte)
    acc_r, _ = eval_acc(m_res, Xte, yte)
    ep = np.arange(1, 7)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(ep, tl_p, "o-", color=C["blue"], label="普通块网络")
    axes[0].plot(ep, tl_r, "s-", color=C["verm"], label="残差块网络")
    axes[0].set_xlabel("训练轮数"); axes[0].set_ylabel("训练损失")
    axes[0].set_title(f"{n_conv} 个卷积层的训练损失")
    axes[0].legend(frameon=False)
    axes[1].plot(ep, va_p, "o-", color=C["blue"], label="普通块网络")
    axes[1].plot(ep, va_r, "s-", color=C["verm"], label="残差块网络")
    axes[1].set_xlabel("训练轮数"); axes[1].set_ylabel("验证准确率")
    axes[1].set_title("验证准确率")
    axes[1].legend(frameon=False)
    fig.suptitle("EXP-6 残差连接：给梯度开一条直通车道，深网络才训得动"
                 "（ResNet 的全部出发点）", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_residual.png"), bbox_inches="tight")
    plt.close(fig)
    STATS["exp6"] = {
        "n_conv_layers": n_conv,
        "train_loss_plain": tl_p, "train_loss_res": tl_r,
        "val_acc_plain": va_p, "val_acc_res": va_r,
        "test_acc_plain": acc_p, "test_acc_res": acc_r,
    }


def exp7_look_where(model, Xte, yte):
    """fig8：类激活高亮 + 背景调暗翻转实验（复现热点机制）。"""
    idx_mel = int(np.where(yte == 1)[0][0])
    idx_nv = int(np.where(yte == 0)[0][0])
    pair = np.array([idx_mel, idx_nv])
    p_pair = model.forward(Xte[pair], cache=True)
    h_last = model._c["flat_h"]               # (2, C, s, s)
    Cc, ss = h_last.shape[1], h_last.shape[2]
    w_fc = model.W[-2]                        # (hidden, C*s*s)
    cam_w = np.abs(w_fc).mean(axis=0).reshape(Cc, ss, ss)
    cams = []
    for i in range(2):
        cam = (h_last[i] * cam_w).sum(axis=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        cams.append(cam)
    # 背景调暗：外圈 4 像素环 ×0.5，中央 24×24 不变
    S = Xte.shape[2]
    r = 4
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    ring = (ii < r) | (ii >= S - r) | (jj < r) | (jj >= S - r)
    mask = np.where(ring, 0.5, 1.0).astype(np.float32)
    Xdark = Xte * mask[None, None]
    p_before = model.forward(Xte)
    p_after = model.forward(Xdark)
    yhat_b = (p_before > 0.5).astype(int)
    yhat_a = (p_after > 0.5).astype(int)
    flip_rate = float((yhat_b != yhat_a).mean())
    acc_before = float((yhat_b == yte).mean())
    acc_after = float((yhat_a == yte).mean())
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    for row, (idx, tag) in enumerate([(idx_mel, "黑色素瘤 mel"),
                                      (idx_nv, "色素痣 nv")]):
        img = Xte[idx, 0]
        cam = np.asarray(Image.fromarray((cams[row] * 255)
                            .astype(np.uint8)).resize((S, S),
                            Image.BILINEAR)) / 255.0
        dark = Xdark[idx, 0]
        pb, pa = p_before[idx], p_after[idx]
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"输入（{tag}）")
        axes[row, 1].imshow(img, cmap="gray")
        axes[row, 1].imshow(cam, cmap="jet", alpha=0.45)
        axes[row, 1].set_title(f"网络看哪里（类激活高亮）\n"
                               f"判为恶性的概率 {pb:.2f}")
        axes[row, 2].imshow(dark, cmap="gray")
        axes[row, 2].set_title(f"外圈调暗到 50% 后\n"
                               f"判为恶性的概率 {pa:.2f}")
        flip = yhat_b[idx] != yhat_a[idx]
        txt = "判断翻转了" if flip else "判断没变"
        axes[row, 3].text(0.5, 0.5, txt, ha="center", va="center",
                          fontsize=15,
                          color=C["verm"] if flip else C["green"])
        for cix in range(4):
            axes[row, cix].axis("off")
    fig.suptitle("EXP-7 网络到底盯着哪片：高亮出来的可能不是病灶，而是周围皮肤\n"
                 f"（外圈调暗后全测试集预测翻转率 {flip_rate:.1%}，"
                 f"准确率 {acc_before:.3f} → {acc_after:.3f}）", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_look_where.png"), bbox_inches="tight")
    plt.close(fig)
    STATS["exp7"] = {
        "dark_factor": 0.5,
        "dark_region": f"外圈 {r} 像素环（中央 {S-2*r}×{S-2*r} 不变）",
        "flip_rate": flip_rate,
        "acc_before_dark": acc_before,
        "acc_after_dark": acc_after,
        "sample_mel_p_before": float(p_before[idx_mel]),
        "sample_mel_p_after": float(p_after[idx_mel]),
        "sample_nv_p_before": float(p_before[idx_nv]),
        "sample_nv_p_after": float(p_after[idx_nv]),
    }


# ===================== 主流程 =====================
def main():
    print("== 梯度数值校验 ==")
    err = grad_check()
    STATS["grad_check_rel_err"] = err
    print(f"   相对误差 {err:.2e}  OK")

    print("== 加载数据 ==")
    ids, y, X96, manifest = load_data()
    print(f"   训练用子集 {len(ids)} 张（mel 1113 全取 + nv 抽样 1000）")

    print("== EXP-0 七类样本 ==")
    exp0_samples(manifest)

    print("== EXP-1 卷积前向 ==")
    exp1_conv_featuremap(ids, y, X96)

    print("== EXP-2 感受野 ==")
    exp2_receptive_field()

    print("== EXP-3 核×步长×padding ==")
    exp3_kernel_stride_padding()

    print("== EXP-4 池化 ==")
    exp4_pooling(ids, y, X96, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                       dtype=np.float32))

    print("== 划分 mel vs nv 二分类（32×32 灰度）==")
    X32 = downscale32(X96)[:, None]          # (N,1,32,32)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X32))
    n = len(X32)
    ntr, nva = int(n * 0.7), int(n * 0.15)
    tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    Xtr, ytr = X32[tr], y[tr]
    Xva, yva = X32[va], y[va]
    Xte, yte = X32[te], y[te]
    STATS["binary_subset"] = {
        "n_total": int(n), "n_train": int(ntr), "n_val": int(nva),
        "n_test": int(len(te)), "img_size": 32, "gray": True,
        "label_def": "mel=1(恶性) / nv=0(良性)", "seed": 42,
    }
    print(f"   训练 {ntr} / 验证 {nva} / 测试 {len(te)}")

    print("== EXP-5 VGG 思想 ==")
    trained5 = exp5_vgg(Xtr, ytr, Xva, yva, Xte, yte)

    print("== EXP-6 残差 ==")
    exp6_residual(Xtr, ytr, Xva, yva, Xte, yte)

    print("== EXP-7 看哪里 ==")
    exp7_look_where(trained5["两层 3×3（VGG）"], Xte, yte)

    with open(os.path.join(ROOT, "stats.json"), "w") as f:
        json.dump(STATS, f, ensure_ascii=False, indent=2)
    print("== stats.json 已落盘 ==")


if __name__ == "__main__":
    main()
