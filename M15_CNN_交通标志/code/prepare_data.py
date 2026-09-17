#!/usr/bin/env python3
"""
M15 数据准备：GTSRB（German Traffic Sign Recognition Benchmark，德国交通标志识别基准）

- 从官方地址下载 GTSRB-Training_fixed.zip（约 179 MB，md5 校验）
- 解析每类的 GT-*.csv，按 ROI 裁出标志区域
- 统一缩放到 32x32 RGB，按类别分层切分训练 / 测试
- 每类上限截断，保证纯 numpy 手写 CNN 在 CPU 上可跑完
- 输出 data/gtsrb32.npz（不入库，脚本可重跑）

GTSRB 简介：2011 年 IJCNN 发布，德国人工智能研究中心（DFKI）与柏林工业大学团队
用真实车载摄像头在德国道路上采集，覆盖不同天气、光照、角度，共 43 类。
"""
import argparse
import csv
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "GTSRB_raw"
ZIP = DATA / "GTSRB-Training_fixed.zip"
NPZ = DATA / "gtsrb32.npz"

URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB-Training_fixed.zip"
)
MD5 = "513f3c79a4c5141765e10e952eaa2478"
IMG_SIZE = 32
SEED = 20260917


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_zip() -> Path:
    if ZIP.exists() and md5_of(ZIP) == MD5:
        print(f"[download] 已有且校验通过：{ZIP}")
        return ZIP
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[download] 开始下载 GTSRB-Training_fixed.zip（约 179 MB）")

    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0:
            pct = min(100.0, done * 100.0 / total)
            print(f"\r[download] {pct:5.1f}%  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")

    urllib.request.urlretrieve(URL, ZIP, reporthook=hook)
    print()
    got = md5_of(ZIP)
    if got != MD5:
        raise RuntimeError(f"md5 校验失败，期望 {MD5}，实际 {got}")
    print("[download] md5 校验通过")
    return ZIP


def ensure_extract(zip_path: Path) -> Path:
    if (RAW / "GTSRB").exists():
        print(f"[extract] 已有解压目录：{RAW / 'GTSRB'}")
        return RAW / "GTSRB"
    print("[extract] 解压中……")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    print("[extract] 解压完成")
    return RAW / "GTSRB"


def load_one_class(class_dir: Path):
    """读取一类的所有样本：按 CSV 的 ROI 裁切并缩放到 32x32。"""
    csv_path = class_dir / f"GT-{class_dir.name}.csv"
    out, labels = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";", skipinitialspace=True):
            x1 = int(row["Roi.X1"]); y1 = int(row["Roi.Y1"])
            x2 = int(row["Roi.X2"]); y2 = int(row["Roi.Y2"])
            cls = int(row["ClassId"])
            img = Image.open(class_dir / row["Filename"]).convert("RGB")
            crop = img.crop((x1, y1, x2, y2))
            small = crop.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            out.append(np.asarray(small, dtype=np.uint8))
            labels.append(cls)
    return np.stack(out), np.array(labels, dtype=np.int64)


def build(cap_train: int, cap_test: int, probe: bool = False):
    zip_path = ensure_zip()
    base = ensure_extract(zip_path)
    # 不同打包版本的目录名不一致，逐个试
    images_root = None
    for cand in (base / "Final_Training" / "Images", base / "Training",
                 base / "Final_Training"):
        if cand.is_dir():
            images_root = cand
            break
    if images_root is None:
        raise RuntimeError(f"在 {base} 下找不到类别目录，请检查解压结构")
    print(f"[data] 类别目录：{images_root}")
    class_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()],
                        key=lambda p: int(p.name))

    rng = np.random.default_rng(SEED)
    xtr, ytr, xte, yte = [], [], [], []
    natural_counts = []

    for cd in class_dirs:
        X, y = load_one_class(cd)
        natural_counts.append(len(y))
        n = len(y)
        idx = rng.permutation(n)
        n_test = max(1, int(round(n * 0.15)))
        n_test = min(n_test, cap_test)
        n_train = min(n - n_test, cap_train)
        te_idx = idx[:n_test]
        tr_idx = idx[n_test:n_test + n_train]
        xtr.append(X[tr_idx]); ytr.append(y[tr_idx])
        xte.append(X[te_idx]); yte.append(y[te_idx])

    X_train = np.concatenate(xtr); y_train = np.concatenate(ytr)
    X_test = np.concatenate(xte); y_test = np.concatenate(yte)

    natural_counts = np.array(natural_counts)
    print(f"[data] 类别数：{len(class_dirs)}")
    print(f"[data] 原始每类样本数：min={natural_counts.min()} "
          f"max={natural_counts.max()} mean={natural_counts.mean():.0f} "
          f"total={natural_counts.sum()}")
    print(f"[data] 取用后 训练 {X_train.shape} / 测试 {X_test.shape}")

    if probe:
        return None

    np.savez_compressed(
        NPZ,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        natural_counts=natural_counts,
    )
    print(f"[data] 已写出：{NPZ}  ({NPZ.stat().st_size / 1e6:.1f} MB)")
    return NPZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-train", type=int, default=300, help="每类训练样本上限")
    ap.add_argument("--cap-test", type=int, default=60, help="每类测试样本上限")
    ap.add_argument("--probe", action="store_true", help="只打印统计，不落盘")
    args = ap.parse_args()
    build(args.cap_train, args.cap_test, probe=args.probe)


if __name__ == "__main__":
    main()
