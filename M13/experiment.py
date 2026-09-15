"""
M13 实验：从零用 NumPy 搭一个 MLP，提前 24 小时预报北京 PM2.5。

数据：UCI Beijing Multi-Site Air-Quality Data (id 501)，Gucheng 单站逐时
      data/PRSA/PRSA_Data_20130301-20170228/PRSA_Data_Gucheng_20130301-20170228.csv
输出：stats.json  —— 文章所有数字的唯一来源
      curves.npz  —— 配图用的损失曲线与测试集预测

一切数学都写在代码里，方便读者逐行对照文章推导：
  - 前向/反向手写，MSE 损失
  - L2 正则的梯度多一项 2*lam*w
  - dropout 用 inverted 方式，训练除以 keep，测试不处理，期望不变
  - 初始化 std 直接决定各层预激活的标准差 sigma_z ~ sqrt(fan_in) * std * sigma_a
"""

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = (ROOT / "data" / "PRSA" / "PRSA_Data_20130301-20170228"
        / "PRSA_Data_Gucheng_20130301-20170228.csv")
STATS = ROOT / "stats.json"
CURVES = ROOT / "curves.npz"

DATA_URL = ("https://archive.ics.uci.edu/static/public/501/"
            "beijing+multi-site+air+quality+data.zip")
CSV_NAME = "PRSA_Data_Gucheng_20130301-20170228.csv"

FEATURES = ["PM2.5", "PM10", "TEMP", "PRES", "DEWP", "WSPM", "RAIN"]
LOOK_BACK = 24          # 用过去 24 小时
HORIZON = 24            # 预报 24 小时后的 PM2.5
SEED = 42


def ensure_data():
    """本地没有就把 UCI 数据拉下来并解开嵌套压缩包。"""
    if DATA.exists():
        return
    print("下载数据 ...", DATA_URL)
    blob = urllib.request.urlopen(DATA_URL, timeout=180).read()
    inner_zip = zipfile.ZipFile(io.BytesIO(blob)).read(
        "PRSA2017_Data_20130301-20170228.zip")
    inner = zipfile.ZipFile(io.BytesIO(inner_zip))
    for name in inner.namelist():
        if name.endswith(CSV_NAME):
            DATA.parent.mkdir(parents=True, exist_ok=True)
            DATA.write_bytes(inner.read(name))
            print("已写入", DATA)
            return
    raise RuntimeError("压缩包里没找到 " + CSV_NAME)


# ----------------------------------------------------------------------------
# 1. 数据读取、缺失处理、窗格化、按时间切分
# ----------------------------------------------------------------------------
def _to_float(v):
    v = v.strip()
    if v in ("", "NA", "NaN", "nan"):
        return np.nan
    return float(v)


def load_raw():
    times, rows = [], []
    with open(DATA, encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {c: header.index(c) for c in FEATURES}
        for line in r:
            if not line:
                continue
            t = (int(line[1]), int(line[2]), int(line[3]), int(line[4]))
            times.append(t)
            rows.append([_to_float(line[idx[c]]) for c in FEATURES])
    return times, np.array(rows, dtype=float)


def ffill(a):
    out = a.copy()
    for j in range(a.shape[1]):
        last = np.nan
        for i in range(a.shape[0]):
            if np.isnan(out[i, j]):
                out[i, j] = last
            else:
                last = out[i, j]
    return out


def make_windows(F, look_back, horizon):
    """X = 过去 look_back 小时的 7 个特征压平，y = horizon 小时后的 PM2.5。"""
    offset = look_back + horizon - 1
    X, y = [], []
    for i in range(len(F) - offset):
        X.append(F[i:i + look_back].reshape(-1))
        y.append(F[i + offset, 0])
    return np.asarray(X), np.asarray(y)


def prepare(look_back=LOOK_BACK, horizon=HORIZON):
    times, raw = load_raw()
    raw = ffill(raw)
    good = ~np.isnan(raw).any(axis=1)
    raw = raw[good]
    times = [t for t, g in zip(times, good) if g]

    n = len(raw)
    n_tr = int(0.70 * n)
    n_va = int(0.85 * n)

    mu = raw[:n_tr].mean(axis=0)
    sd = raw[:n_tr].std(axis=0) + 1e-8
    Fn = (raw - mu) / sd

    X, y_std = make_windows(Fn, look_back, horizon)
    offset = look_back + horizon - 1
    cut_tr = n_tr - offset
    cut_va = n_va - offset
    Xtr, ytr = X[:cut_tr], y_std[:cut_tr]
    Xva, yva = X[cut_tr:cut_va], y_std[cut_tr:cut_va]
    Xte, yte = X[cut_va:], y_std[cut_va:]

    meta = {
        "n_rows": int(n),
        "pm_mu": round(float(mu[0]), 3),
        "pm_sd": round(float(sd[0]), 3),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "t_start": "%04d-%02d-%02d %02d:00" % times[0],
        "t_end": "%04d-%02d-%02d %02d:00" % times[-1],
        "look_back": look_back,
        "horizon": horizon,
        "n_features": len(FEATURES),
    }
    return (Xtr, ytr), (Xva, yva), (Xte, yte), (mu[0], sd[0]), meta


# ----------------------------------------------------------------------------
# 2. 从零实现 MLP
# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, lr=1e-2, init="xavier", init_std=0.1,
                 dropout=0.0, l2=0.0, seed=SEED):
        self.sizes = sizes
        self.lr = lr
        self.dropout = dropout
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            if init == "xavier":
                std = np.sqrt(1.0 / fan_in)
            elif init == "he":
                std = np.sqrt(2.0 / fan_in)
            else:
                std = init_std
            self.W.append(self.rng.normal(0.0, std, (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    def forward(self, X, training=False):
        self.zs, self.acts, self.masks = [], [], []
        a = X
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            self.zs.append(z)
            if i < L - 1:
                h = np.tanh(z)
                mask = None
                if training and self.dropout > 0:
                    keep = 1.0 - self.dropout
                    mask = (self.rng.random(h.shape) < keep) / keep
                    h = h * mask
                self.masks.append(mask)
                self.acts.append(h)
                a = h
            else:
                a = z
        return a

    def train_epoch(self, X, y, batch=256):
        n = len(X)
        idx = self.rng.permutation(n)
        L = len(self.W)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            xb, yb = X[bi], y[bi]
            pred = self.forward(xb, training=True)
            err = pred.ravel() - yb
            g = (2.0 / len(bi)) * err.reshape(-1, 1)
            gW = [None] * L
            gb = [None] * L
            for i in reversed(range(L)):
                prev_a = xb if i == 0 else self.acts[i - 1]
                gW[i] = prev_a.T @ g
                gb[i] = g.sum(axis=0)
                if i > 0:
                    dz = g @ self.W[i].T
                    t = np.tanh(self.zs[i - 1])
                    dz = dz * (1.0 - t ** 2)
                    if self.masks[i - 1] is not None:
                        dz = dz * self.masks[i - 1]
                    g = dz
            for i in range(L):
                gradW = gW[i]
                if self.l2 > 0:
                    gradW = gradW + 2.0 * self.l2 * self.W[i]
                self.W[i] -= self.lr * gradW
                self.b[i] -= self.lr * gb[i]

    def predict(self, X, batch=4096):
        out = []
        for s in range(0, len(X), batch):
            out.append(self.forward(X[s:s + batch], training=False))
        return np.concatenate(out, axis=0).ravel()

    def weight_norm(self):
        return float(np.sqrt(sum(np.sum(w ** 2) for w in self.W)))

    def act_stats(self, X):
        self.forward(X[:512], training=False)
        return [float(np.std(z)) for z in self.zs]


def fit(model, Xtr, ytr, Xva, yva, epochs, record=False):
    tr_curve, va_curve = [], []
    for _ in range(epochs):
        model.train_epoch(Xtr, ytr)
        if record:
            tr_curve.append(float(np.mean((model.predict(Xtr) - ytr) ** 2)))
            va_curve.append(float(np.mean((model.predict(Xva) - yva) ** 2)))
    return tr_curve, va_curve


def metrics(pred_std, y_std, pm_mu, pm_sd):
    pred = pred_std * pm_sd + pm_mu
    true = y_std * pm_sd + pm_mu
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot)
    mae = float(np.mean(np.abs(true - pred)))
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "r2": round(r2, 4)}


# ----------------------------------------------------------------------------
# 3. 跑全部实验
# ----------------------------------------------------------------------------
def main():
    ensure_data()
    (Xtr, ytr), (Xva, yva), (Xte, yte), (pm_mu, pm_sd), meta = prepare()
    n_feat = Xtr.shape[1]
    SIZES = [n_feat, 64, 32, 1]
    LAST = (LOOK_BACK - 1) * len(FEATURES) + 0      # 窗口里最后一个 PM2.5
    res = {"meta": meta, "arch": SIZES}
    curves = {}
    print("数据", meta["n_train"], meta["n_val"], meta["n_test"], "horizon", HORIZON)

    # 基线：持续性（拿窗口最后一个 PM2.5 当 24 小时后的预测）
    res["baseline_persistence"] = {
        "val": metrics(Xva[:, LAST], yva, pm_mu, pm_sd),
        "test": metrics(Xte[:, LAST], yte, pm_mu, pm_sd),
    }
    print("基线", res["baseline_persistence"]["test"])

    # 对照：horizon=1 时的持续性 vs 模型，用来说明为什么改预报一天后
    (Xtr1, ytr1), (Xva1, yva1), (Xte1, yte1), _, _ = prepare(horizon=1)
    m1 = MLP([Xtr1.shape[1], 64, 32, 1], lr=1e-2, init="xavier")
    fit(m1, Xtr1, ytr1, Xva1, yva1, epochs=60)
    LAST1 = (LOOK_BACK - 1) * len(FEATURES) + 0
    res["horizon_compare"] = {
        "h1_persistence": metrics(Xte1[:, LAST1], yte1, pm_mu, pm_sd),
        "h1_model": metrics(m1.predict(Xte1), yte1, pm_mu, pm_sd),
    }
    print("H=1 基线", res["horizon_compare"]["h1_persistence"],
          "模型", res["horizon_compare"]["h1_model"])

    # 实验 A：学习率扫描（含过大发散）
    lrs = [1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]
    res["lr_sweep"] = {}
    for lr in lrs:
        m = MLP(SIZES, lr=lr, init="xavier")
        trc, vac = fit(m, Xtr, ytr, Xva, yva, epochs=40, record=True)
        div = bool((not np.isfinite(vac[-1])) or vac[-1] > 5.0)
        curves[f"lr_{lr}_train"] = np.array(trc)
        curves[f"lr_{lr}_val"] = np.array(vac)
        res["lr_sweep"][str(lr)] = {
            "final_train_mse": round(float(np.nan_to_num(trc[-1], nan=-1)), 4),
            "final_val_mse": round(float(np.nan_to_num(vac[-1], nan=-1)), 4),
            "diverged": div,
            "val_curve": [round(float(np.nan_to_num(v, nan=-1)), 3) for v in vac[::4]],
        }
        print("lr", lr, "val", res["lr_sweep"][str(lr)]["final_val_mse"], "div", div)

    # 实验 B：初始化 std 扫描（浅层网）
    stds = [0.01, 0.05, 0.1, 0.5]
    res["init_std"] = {}
    for s in stds:
        m = MLP(SIZES, lr=1e-2, init="normal", init_std=s)
        a0 = m.act_stats(Xtr)
        trc, vac = fit(m, Xtr, ytr, Xva, yva, epochs=40, record=True)
        res["init_std"][str(s)] = {
            "z_std_per_layer": [round(v, 3) for v in a0],
            "final_train_mse": round(trc[-1], 4),
            "final_val_mse": round(vac[-1], 4),
        }
        print("std", s, "z_std", res["init_std"][str(s)]["z_std_per_layer"],
              "val", round(vac[-1], 4))

    # 实验 C：深层网的初始化（同时看逐层坍缩与饱和）
    DEEP = [n_feat, 64, 64, 64, 64, 32, 1]
    res["init_depth"] = {}
    for name, kw in [("std_0.01", dict(init="normal", init_std=0.01)),
                     ("xavier", dict(init="xavier")),
                     ("he", dict(init="he")),
                     ("std_0.5", dict(init="normal", init_std=0.5))]:
        m = MLP(DEEP, lr=1e-2, **kw)
        res["init_depth"][name] = {
            "z_std_per_layer": [round(v, 4) for v in m.act_stats(Xtr)],
        }
        print("deep", name, res["init_depth"][name]["z_std_per_layer"])

    # 实验 D：dropout 扫描
    res["dropout"] = {}
    for p in [0.0, 0.2, 0.5]:
        m = MLP(SIZES, lr=1e-2, init="xavier", dropout=p)
        trc, vac = fit(m, Xtr, ytr, Xva, yva, epochs=60, record=True)
        curves[f"drop_{p}_train"] = np.array(trc)
        curves[f"drop_{p}_val"] = np.array(vac)
        res["dropout"][str(p)] = {
            "final_train_mse": round(trc[-1], 4),
            "final_val_mse": round(vac[-1], 4),
            "gap": round(vac[-1] - trc[-1], 4),
            "val": metrics(m.predict(Xva), yva, pm_mu, pm_sd),
        }
        print("dropout", p, "gap", res["dropout"][str(p)]["gap"])

    # 实验 E：L2 权重衰减扫描
    res["l2"] = {}
    for lam in [0.0, 1e-4, 1e-3, 1e-2]:
        m = MLP(SIZES, lr=1e-2, init="xavier", l2=lam)
        trc, vac = fit(m, Xtr, ytr, Xva, yva, epochs=60, record=True)
        res["l2"][str(lam)] = {
            "final_train_mse": round(trc[-1], 4),
            "final_val_mse": round(vac[-1], 4),
            "weight_norm": round(m.weight_norm(), 4),
            "val": metrics(m.predict(Xva), yva, pm_mu, pm_sd),
        }
        print("l2", lam, "val", round(vac[-1], 4),
              "w_norm", res["l2"][str(lam)]["weight_norm"])

    # 最终配置：跑久一点，报测试集
    best = MLP(SIZES, lr=1e-2, init="xavier", dropout=0.2, l2=1e-4)
    trc, vac = fit(best, Xtr, ytr, Xva, yva, epochs=200, record=True)
    curves["best_train"] = np.array(trc)
    curves["best_val"] = np.array(vac)
    pred_te = best.predict(Xte)
    res["best"] = {
        "config": "lr=1e-2, xavier, dropout=0.2, l2=1e-4, 200 epochs",
        "train_mse": round(trc[-1], 4),
        "val_mse": round(vac[-1], 4),
        "test": metrics(pred_te, yte, pm_mu, pm_sd),
        "weight_norm": round(best.weight_norm(), 4),
        "persistence_test": res["baseline_persistence"]["test"],
    }
    # 存一段测试集预测用于画图
    k = min(720, len(yte))
    curves["best_test_true"] = (yte[:k] * pm_sd + pm_mu)
    curves["best_test_pred"] = (pred_te[:k] * pm_sd + pm_mu)
    curves["best_test_persist"] = (Xte[:k, LAST] * pm_sd + pm_mu)
    print("best", res["best"])

    STATS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(CURVES, **curves)
    print("written", STATS.name, CURVES.name)


if __name__ == "__main__":
    main()
