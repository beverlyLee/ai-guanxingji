#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV4 图像处理实战篇 —— 三个项目的可复现实验脚本。
数据来源：MNIST（真实公开，ossci 镜像）、OpenCV 官方示例图（公开）。
停车位场景为脚本生成的示意图（零依赖、可复现），文中标注真实公开集 PKLot 的替换方式。
产出：../stats.json（文章所有数字的唯一来源）+ ../figures_data/*.npz（配图数据）。
"""
import os
import json
import gzip
import math
import struct
import urllib.request
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGDATA = os.path.join(ROOT, "figures_data")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGDATA, exist_ok=True)

SEED = 20260919
rng = np.random.default_rng(SEED)

STATS = {}


# ----------------------------------------------------------------------------
# 通用：带重试的下载
# ----------------------------------------------------------------------------
def download(url, dest, retries=6, timeout=60):
    if os.path.exists(dest):
        return dest
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cv4-exp"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                with open(dest, "wb") as f:
                    f.write(r.read())
            return dest
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(f"download failed: {url} -> {last}")


# ----------------------------------------------------------------------------
# 项目一：MNIST（真实公开数据）
# ----------------------------------------------------------------------------
def parse_idx(path, kind):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if kind == "images":
            nr, nc = struct.unpack(">II", f.read(8))
            buf = f.read(n * nr * nc)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, nr, nc)
        else:
            buf = f.read(n)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n)
    return arr


def load_mnist():
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "tr_img": "train-images-idx3-ubyte.gz",
        "tr_lbl": "train-labels-idx1-ubyte.gz",
        "te_img": "t10k-images-idx3-ubyte.gz",
        "te_lbl": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {}
    for k, fn in files.items():
        paths[k] = download(base + fn, os.path.join(DATA, fn))
    tr_img = parse_idx(paths["tr_img"], "images").astype(np.float32)
    tr_lbl = parse_idx(paths["tr_lbl"], "labels")
    te_img = parse_idx(paths["te_img"], "images").astype(np.float32)
    te_lbl = parse_idx(paths["te_lbl"], "labels")
    return tr_img, tr_lbl, te_img, te_lbl


def train_mnist_clf(tr_img, tr_lbl, te_img, te_lbl):
    from sklearn.neural_network import MLPClassifier
    STATS["mnist"] = {
        "n_train": int(tr_img.shape[0]),
        "n_test": int(te_img.shape[0]),
        "img_h": int(tr_img.shape[1]),
        "img_w": int(tr_img.shape[2]),
    }
    cls, counts = np.unique(tr_lbl, return_counts=True)
    STATS["mnist"]["class_counts"] = {int(c): int(n) for c, n in zip(cls, counts)}

    idx = rng.choice(tr_img.shape[0], 20000, replace=False)
    Xtr = tr_img[idx].reshape(-1, 784) / 255.0
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, alpha=1e-4,
                        batch_size=512, random_state=SEED, early_stopping=False)
    clf.fit(Xtr, tr_lbl[idx])
    Xte = te_img.reshape(-1, 784) / 255.0
    acc = float(clf.score(Xte, te_lbl))
    STATS["mnist"]["mlp_test_acc"] = round(acc, 4)

    samples = []
    for d in range(10):
        s = np.where(tr_lbl == d)[0][:5]
        samples.append(tr_img[s])
    samples = np.array(samples)
    np.savez(os.path.join(FIGDATA, "mnist_samples.npz"), samples=samples)
    return clf


def make_credit_card(te_img, te_lbl, clf, n_digits=16):
    """把真实 MNIST 数字（反相成深色）贴成一张均匀浅色卡面，记录真值标签。"""
    W, Hh = 760, 480
    card = np.full((Hh, W), 238, dtype=np.float32)
    yy, xx = np.mgrid[0:Hh, 0:W]
    card = card - yy.astype(np.float32) * 0.05 + rng.normal(0, 2.5, (Hh, W))
    card = np.clip(card, 150, 255).astype(np.uint8)

    chosen, gt, used = [], [], set()
    attempts = 0
    while len(chosen) < n_digits and attempts < 8000:
        attempts += 1
        i = int(rng.integers(0, len(te_img)))
        if i in used:
            continue
        used.add(i)
        chosen.append(te_img[i]); gt.append(int(te_lbl[i]))

    groups, per_group = 4, 4
    gx0, gy = 90, Hh // 2
    span = W - 2 * gx0
    dx_group = span / groups
    dx_digit = dx_group / per_group
    ds = 32   # 必须小于 dx_digit(~36)，否则相邻数字重叠连成一块
    boxes = []
    for order in range(n_digits):
        g = order // per_group
        p = order % per_group
        cx = int(gx0 + g * dx_group + p * dx_digit + dx_digit / 2)
        cy = gy
        d = chosen[order]
        dimg = cv2.resize(d, (ds, ds)).astype(np.int16)   # 黑字（与 MNIST 训练格式一致）
        x0, y0 = cx - ds // 2, cy - ds // 2
        region = card[y0:y0 + ds, x0:x0 + ds].astype(np.int16)
        region = np.minimum(region, dimg)
        card[y0:y0 + ds, x0:x0 + ds] = region.astype(np.uint8)
        boxes.append((x0, y0, ds, ds))

    # ---- CV1 定位流水线 ----
    otsu_t, bin_img = cv2.threshold(card, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_clean = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in contours:
        a = cv2.contourArea(c)
        if 60 < a < 3500:
            x, y, w, h = cv2.boundingRect(c)
            cand.append((x, y, w, h))
    cand_sorted = sorted(cand, key=lambda b: (b[1], b[0]))

    feats = []
    for (x, y, w, h) in cand_sorted:
        crop = card[y:y + h, x:x + w]
        crop = cv2.resize(crop, (28, 28)).astype(np.float32) / 255.0
        feats.append(crop.flatten())
    rec = clf.predict(np.array(feats)) if feats else np.array([])
    n_det = len(rec)
    n_match = sum(1 for a, b in zip(rec, gt[:n_det]) if int(a) == b)
    card_acc = n_match / n_det if n_det else 0.0
    wrong = [{"pos": int(i), "gt": int(gt[i]), "pred": int(rec[i]),
              "box_w": int(cand_sorted[i][2]), "box_h": int(cand_sorted[i][3])}
             for i in range(min(len(rec), len(gt))) if int(rec[i]) != gt[i]]
    ar = [round(b[2] / b[3], 3) for b in cand_sorted]

    STATS["credit_card"] = {
        "n_digits_truth": n_digits,
        "n_digits_detected": int(n_det),
        "n_recognized_correct": int(n_match),
        "card_accuracy": round(float(card_acc), 4),
        "wrong_digits": wrong,
        "box_ar_min": min(ar) if ar else 0.0,
        "box_ar_max": max(ar) if ar else 0.0,
        "box_ar_mean": round(float(np.mean(ar)), 3) if ar else 0.0,
    }
    STATS["credit_card"]["otsu_threshold"] = int(float(np.asarray(otsu_t).ravel()[0]))
    panel = card[gy - 60:gy + 60, gx0 - 30:W - 30]
    hist = cv2.calcHist([panel], [0], None, [256], [0, 256]).flatten()
    np.savez(os.path.join(FIGDATA, "card.npz"),
             card=card, det_boxes=np.array(cand_sorted), hist=hist.astype(np.float32),
             otsu_t=int(float(np.asarray(otsu_t).ravel()[0])))
    return card, cand_sorted


# ----------------------------------------------------------------------------
# 项目二：停车位识别（脚本生成的示意图，零依赖可复现）
# ----------------------------------------------------------------------------
def make_parking_frame(n_rows=2, n_cols=6, occ_prob=0.5):
    Hh, W = 420, 720
    img = np.clip(110 + rng.normal(0, 6, (Hh, W)), 60, 180).astype(np.uint8)
    bay_w = W / n_cols
    bay_h = (Hh - 60) / n_rows
    y0 = 30
    gt_occ = 0
    for r in range(n_rows):
        for c in range(n_cols):
            x = int(c * bay_w) + 8
            y = int(y0 + r * bay_h) + 8
            w = int(bay_w) - 16
            h = int(bay_h) - 16
            cv2.rectangle(img, (x, y), (x + w, y + h), 200, 2)
            if rng.random() < occ_prob:
                gt_occ += 1
                m = 8
                region = img[y + m:y + h - m, x + m:x + w - m].astype(np.int16) - 70
                img[y + m:y + h - m, x + m:x + w - m] = np.clip(region, 0, 255).astype(np.uint8)
    return img, gt_occ


def parking_experiment(n_frames=30):
    tp = fp = fn = 0
    frame_dets = []
    sample_img = sample_overlay = None
    bin_first = opened_first = None
    for i in range(n_frames):
        img, gt_occ = make_parking_frame()
        # CV1 工具链：全局 Otsu 阈值（占用块是均匀暗色，自适应阈值反而检不出）+ 形态学开运算 + 轮廓
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        det = sum(1 for c in contours if cv2.contourArea(c) > 600)
        if det >= gt_occ:
            tp += gt_occ; fp += (det - gt_occ)
        else:
            tp += det; fn += (gt_occ - det)
        frame_dets.append((gt_occ, det))
        if i == 0:
            sample_img = img
            overlay = img.copy()
            for c in contours:
                if cv2.contourArea(c) > 600:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), 255, 2)
            sample_overlay = overlay
            bin_first, opened_first = bin_img, opened
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    STATS["parking"] = {
        "n_frames": n_frames, "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0, 4),
        "n_rows": 2, "n_cols": 6, "occ_prob": 0.5, "area_thresh": 600,
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt_total": tp + fn, "n_det_total": tp + fp,
        "bays_per_frame": 12,
        "bay_box_w": 104, "bay_box_h": 164,
        "occ_block_w": 88, "occ_block_h": 148, "occ_block_area": 88 * 148,
        "otsu_note": "全局 Otsu 阈值（THRESH_BINARY_INV），占用块为均匀暗色故不用自适应阈值",
    }
    np.savez(os.path.join(FIGDATA, "parking_morph.npz"), bin=bin_first, opened=opened_first)
    np.savez(os.path.join(FIGDATA, "parking.npz"),
             sample_img=sample_img, sample_overlay=sample_overlay, frame_dets=np.array(frame_dets))
    return


# ----------------------------------------------------------------------------
# 项目三：全景拼接（真实公开底图 + 已知单应变换）
# ----------------------------------------------------------------------------
def load_base_image():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"
    dest = os.path.join(DATA, "building.jpg")
    try:
        download(url, dest, retries=8, timeout=90)
        img = cv2.imread(dest)
        if img is not None and img.size > 1000:
            return img
    except Exception as e:  # noqa
        print("base image download failed:", e)
    h, w = 360, 520
    img = np.zeros((h, w, 3), dtype=np.uint8)
    r2 = np.random.default_rng(42)
    for _ in range(400):
        x, y = r2.integers(0, w), r2.integers(0, h)
        cv2.circle(img, (int(x), int(y)), r2.integers(3, 12),
                   (r2.integers(80, 255), r2.integers(80, 255), r2.integers(80, 255)), -1)
    for _ in range(40):
        x1, y1 = r2.integers(0, w), r2.integers(0, h)
        x2, y2 = r2.integers(0, w), r2.integers(0, h)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    return img


def ransac_iters(w, s=4, p=0.99):
    """达到置信度 p 所需的 RANSAC 迭代次数：N = log(1-p) / log(1 - w^s)。"""
    if not (0.0 < w < 1.0):
        return None
    denom = math.log(1.0 - w ** s)
    if denom == 0.0:
        return None
    return int(math.ceil(math.log(1.0 - p) / denom))


def panorama_experiment():
    img1 = load_base_image()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h, w = gray1.shape
    ang = np.deg2rad(5.0)
    c, s = np.cos(ang), np.sin(ang)
    cx, cy = w / 2.0, h / 2.0
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array([[1, 0, 50], [0, 1, 25], [0, 0, 1]], dtype=np.float64)
    S = np.array([[1.03, 0, 0], [0, 1.03, 0], [0, 0, 1]], dtype=np.float64)
    H_gt = (T @ R @ S).astype(np.float64)
    A1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    A2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    H_gt = (A2 @ H_gt @ A1).astype(np.float64)

    img2 = cv2.warpPerspective(img1, H_gt, (w, h))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.8 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_rec, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H_rec is not None and inliers > 0:
        proj = cv2.perspectiveTransform(pts1, H_rec)
        errs = np.sqrt(((proj - pts2) ** 2).sum(axis=2)).flatten()
        errs = errs[mask.flatten() == 1]
        mean_reproj = float(errs.mean()) if len(errs) else 0.0
        ys, xs = np.mgrid[0:h:20, 0:w:20]
        grid = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs.ravel())], axis=1).astype(np.float64)
        p_gt = (H_gt @ grid.T).T[:, :2] / (H_gt @ grid.T).T[:, 2:3]
        p_rec = (H_rec @ grid.T).T[:, :2] / (H_rec @ grid.T).T[:, 2:3]
        h_err = float(np.sqrt(((p_gt - p_rec) ** 2).sum(axis=1)).mean())
    else:
        mean_reproj = 0.0; h_err = 0.0

    stitched = img1.copy()
    warped = cv2.warpPerspective(img2, np.linalg.inv(H_rec) if H_rec is not None else np.eye(3), (w, h))
    gw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, msk = cv2.threshold(gw, 1, 255, cv2.THRESH_BINARY)
    stitched[msk == 255] = warped[msk == 255]

    ir = float(inliers / len(good)) if good else 0.0
    ws = [0.3, 0.4, 0.5, 0.6, 0.8, round(ir, 4)]
    STATS["panorama"] = {
        "n_keypoints_img1": int(len(kp1)), "n_keypoints_img2": int(len(kp2)),
        "n_matches_ratio_test": int(len(good)),
        "n_ransac_inliers": inliers,
        "inlier_ratio": round(ir, 4),
        "mean_reproj_error_px": round(mean_reproj, 3),
        "recovered_vs_gt_homography_err_px": round(h_err, 3),
        "ransac_threshold_px": 5.0, "rotation_deg": 5.0, "scale": 1.03, "translation_px": [50, 25],
        "ransac_conf_p": 0.99, "ransac_sample_size": 4,
        "ransac_iters_table": [[w, ransac_iters(w)] for w in ws],
        "ratio_test_thresh": 0.8,
        "sift_descriptor_dim": 128,
        "homography_dof": 8,
    }
    kp1_pt = np.array([k.pt for k in kp1]); kp2_pt = np.array([k.pt for k in kp2])
    match_pts = np.stack([pts1.reshape(-1, 2), pts2.reshape(-1, 2)], axis=1)
    inlier_mask = mask.flatten() == 1 if mask is not None else np.zeros(len(good), dtype=bool)
    np.savez(os.path.join(FIGDATA, "panorama.npz"),
             img1=img1, img2=img2, stitched=stitched, kp1_pt=kp1_pt, kp2_pt=kp2_pt,
             match_pts=match_pts, inlier_mask=inlier_mask, H_rec=H_rec if H_rec is not None else np.eye(3))


def main():
    print(">> MNIST + train clf ...")
    tr_img, tr_lbl, te_img, te_lbl = load_mnist()
    clf = train_mnist_clf(tr_img, tr_lbl, te_img, te_lbl)
    print("   mlp test acc:", STATS["mnist"]["mlp_test_acc"])
    print(">> credit card ...")
    make_credit_card(te_img, te_lbl, clf)
    print("   card:", STATS["credit_card"])
    print(">> parking ...")
    parking_experiment(30)
    print("   parking:", STATS["parking"])
    print(">> panorama ...")
    panorama_experiment()
    print("   panorama:", STATS["panorama"])
    with open(os.path.join(ROOT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(STATS, f, ensure_ascii=False, indent=2)
    print("saved stats.json")


if __name__ == "__main__":
    main()
