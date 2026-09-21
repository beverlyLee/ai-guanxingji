"""
CV2 OpenCV 边缘与轮廓 — 用 scikit-image 公共测试图做真实实验，产出 figures + stats.json
运行环境：受管 python venv（需 opencv-python-headless + scikit-image + numpy + matplotlib + scipy）
所有数字真实可复现，写入 ../stats.json

知识点覆盖：
  Sobel/Scharr/Laplacian 梯度算子、Canny 五步（高斯→Sobel→非极大值抑制→双阈值→滞后连接）、
  图像金字塔、轮廓检测与近似、模板匹配。
数据集：scikit-image 自带 coins（数硬币主角）+ camera（演示梯度四算子对比）。零下载、零敏感、可复现。
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage.data as skd
import cv2
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "..", "figures")
DATA = os.path.join(ROOT, "..", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

stats = {}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans"})

# ---------- 数据 ----------
camera = skd.camera()   # 灰度 512x512
coins = skd.coins()     # 灰度 303x384
stats["shapes"] = {"camera": list(camera.shape), "coins": list(coins.shape)}
stats["coins_dtype"] = str(coins.dtype)
stats["coins_range"] = [int(coins.min()), int(coins.max())]
stats["camera_range"] = [int(camera.min()), int(camera.max())]

# ---------- 算子核（写入 stats，文章里要讲清 3x3 数字来源） ----------
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = sobel_x.T
scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
scharr_y = scharr_x.T
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
stats["kernel_sobel_x"] = sobel_x.tolist()
stats["kernel_sobel_y"] = sobel_y.tolist()
stats["kernel_scharr_x"] = scharr_x.tolist()
stats["kernel_laplacian"] = laplacian.tolist()

# ---------- 1. Sobel 三件套（Gx / Gy / 幅度 / 方向） ----------
gx = cv2.Sobel(camera, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(camera, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(gx ** 2 + gy ** 2)
direction = np.degrees(np.arctan2(gy, gx)) % 180   # 0..180 梯度方向
stats["sobel_mag_max"] = round(float(mag.max()), 2)


def norm01(x):
    x = np.abs(x)
    return x / x.max() if x.max() > 0 else x


fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for a, im, t in zip(axes, [norm01(gx), norm01(gy), norm01(mag), direction / 180.0],
                    ["Sobel Gx (vertical edge)", "Sobel Gy (horizontal edge)",
                     "gradient magnitude", "gradient direction 0-180"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_sobel.png")); plt.close(fig)

# ---------- 2. Scharr / Laplacian 与 Sobel 对比 ----------
sx_s = cv2.Scharr(camera, cv2.CV_64F, 1, 0)
sy_s = cv2.Scharr(camera, cv2.CV_64F, 0, 1)
mag_s = np.sqrt(sx_s ** 2 + sy_s ** 2)
lap = cv2.Laplacian(camera, cv2.CV_64F, ksize=3)
stats["scharr_mag_max"] = round(float(mag_s.max()), 2)
stats["laplacian_max"] = round(float(np.abs(lap).max()), 2)

# 边缘像素占比（固定相对阈值，便于横向比较）
sobel_ratio = float((mag > 0.15 * mag.max()).mean())
scharr_ratio = float((mag_s > 0.15 * mag_s.max()).mean())
lap_ratio = float((np.abs(lap) > 0.10 * np.abs(lap).max()).mean())
stats["edge_ratio_sobel"] = round(sobel_ratio, 4)
stats["edge_ratio_scharr"] = round(scharr_ratio, 4)
stats["edge_ratio_laplacian"] = round(lap_ratio, 4)


def ncomp(b):
    lab, n = ndimage.label(b)
    return int(n)


bw_sobel = (mag > 0.15 * mag.max()).astype(np.uint8)
bw_scharr = (mag_s > 0.15 * mag_s.max()).astype(np.uint8)
bw_lap = (np.abs(lap) > 0.10 * np.abs(lap).max()).astype(np.uint8)
stats["cc_sobel"] = ncomp(bw_sobel)
stats["cc_scharr"] = ncomp(bw_scharr)
stats["cc_laplacian"] = ncomp(bw_lap)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(axes, [norm01(mag), norm01(mag_s), norm01(lap)],
                    ["Sobel magnitude", "Scharr magnitude", "Laplacian (2nd-order)"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_operators.png")); plt.close(fig)

# ---------- 3. Canny 五步（手搓，逐步出图，突出非极大值抑制与滞后连接） ----------
# Step 1 高斯模糊
blurred = cv2.GaussianBlur(camera, (5, 5), 1.4)
# Step 2 Sobel 幅度 + 方向（在模糊图上）
gx2 = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gy2 = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
mag2 = np.sqrt(gx2 ** 2 + gy2 ** 2)
ang2 = np.degrees(np.arctan2(gy2, gx2)) % 180
# Step 3 非极大值抑制（4 扇区向量化）
A = ang2
E = ((A >= 0) & (A < 22.5)) | (A >= 157.5)
NE = (A >= 22.5) & (A < 67.5)
N = (A >= 67.5) & (A < 112.5)
NW = (A >= 112.5) & (A < 157.5)
m = np.pad(mag2, 1, mode="edge")
left = m[1:-1, :-2]; right = m[1:-1, 2:]
up = m[:-2, 1:-1]; down = m[2:, 1:-1]
ulup = m[:-2, :-2]; drdn = m[2:, 2:]
uldn = m[:-2, 2:]; drup = m[2:, :-2]
neigh_max = np.zeros_like(mag2)
neigh_max = np.where(E, np.maximum(left, right), neigh_max)
neigh_max = np.where(NE, np.maximum(ulup, drdn), neigh_max)
neigh_max = np.where(N, np.maximum(up, down), neigh_max)
neigh_max = np.where(NW, np.maximum(uldn, drup), neigh_max)
nms = mag2.copy()
nms[mag2 < neigh_max] = 0
# Step 4 双阈值（强/弱）
high = 0.18 * mag2.max()
low = 0.09 * mag2.max()
strong = nms > high
weak = (nms >= low) & (nms <= high)
# Step 5 滞后连接：把连到强边缘的弱边缘保留
edges = strong.astype(np.uint8)
for _ in range(50):
    dilated = ndimage.binary_dilation(edges) & weak
    if dilated.sum() == 0:
        break
    edges = edges | dilated
# 计数（NMS 前 / NMS 后 / 强 / 弱 / 滞后后）
raw_above_low = int((mag2 >= low).sum())
nms_above_low = int((nms >= low).sum())
strong_n = int(strong.sum())
weak_n = int(weak.sum())
final_n = int(edges.sum())
stats["canny_low_thr"] = round(float(low), 2)
stats["canny_high_thr"] = round(float(high), 2)
stats["canny_raw_above_low"] = raw_above_low
stats["canny_after_nms"] = nms_above_low
stats["canny_strong"] = strong_n
stats["canny_weak"] = weak_n
stats["canny_final"] = final_n
stats["canny_nms_reduction"] = round(raw_above_low / nms_above_low, 2) if nms_above_low else None
stats["canny_final_ratio"] = round(final_n / mag2.size, 4)
# 对比 cv2 官方 Canny
cv_canny = cv2.Canny(camera, int(low), int(high))
stats["canny_opencv_final"] = int((cv_canny > 0).sum())

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
panels = [
    (camera, "1. original camera"),
    (norm01(blurred), "2. gaussian blur (sigma=1.4)"),
    (norm01(mag2), "3. sobel magnitude"),
    (norm01(nms), "4. after non-max suppression"),
    (np.where(strong | weak, 1.0, 0.0), "5. dual threshold (strong+weak)"),
    (edges.astype(float), "6. hysteresis -> final edges"),
]
for a, (im, t) in zip(axes.ravel(), panels):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_canny.png")); plt.close(fig)

# ---------- 4. 图像金字塔（coins） ----------
pyr = [coins.astype(np.float32)]
for _ in range(4):
    pyr.append(cv2.pyrDown(pyr[-1]))
stats["pyramid_levels"] = []
for k, p in enumerate(pyr):
    stats["pyramid_levels"].append({"level": k, "shape": [int(p.shape[0]), int(p.shape[1])],
                                    "mean": round(float(p.mean()), 2),
                                    "std": round(float(p.std()), 2)})
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for a, p, k in zip(axes, pyr, range(len(pyr))):
    a.imshow(p, cmap="gray"); a.set_title(f"level {k}\n{p.shape[1]}x{p.shape[0]}"); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_pyramid.png")); plt.close(fig)

# ---------- 5. 轮廓检测与近似（coins 数硬币） ----------
# Otsu 二值化：硬币比背景亮 -> 255
_, otsu = cv2.threshold(coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 形态学：闭运算填洞 + 开运算去小噪点
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kern), cv2.MORPH_OPEN, kern)
cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
stats["contour_raw_count"] = len(cnts)


def circularity(c):
    a = cv2.contourArea(c)
    if a <= 0:
        return 0.0
    p = cv2.arcLength(c, True)
    return 4 * np.pi * a / (p ** 2) if p > 0 else 0.0


# 候选筛选：去掉极小碎片(<200)、巨大背景块(>5% 图像)、保留圆形块(circularity>=0.6)
cands = []
for c in cnts:
    a = cv2.contourArea(c)
    if a < 200 or a > 0.05 * coins.size:
        continue
    if circularity(c) < 0.6:
        continue
    cands.append(c)
cands.sort(key=cv2.contourArea, reverse=True)
stats["contour_candidates"] = len(cands)

# 7 枚硬币 = 7 个最大的圆形块（其余是硬币/桌面的反光亮斑，面积更小且发散）
COIN_N = 7
coin_cnts = cands[:COIN_N]
stats["coins_detected"] = len(coin_cnts)

# 逐个记录轮廓特征
coin_features = []
for i, c in enumerate(coin_cnts):
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    circ = circularity(c)
    (ccx, ccy), cr = cv2.minEnclosingCircle(c)
    hull = cv2.convexHull(c)
    solid = a / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    coin_features.append({"idx": i, "area": round(float(a), 1), "perimeter": round(float(p), 1),
                          "circularity": round(float(circ), 4), "enclosing_r": round(float(cr), 1),
                          "solidity": round(float(solid), 4)})
stats["coin_features"] = coin_features
stats["coin_area_min"] = round(min(f["area"] for f in coin_features), 1)
stats["coin_area_max"] = round(max(f["area"] for f in coin_features), 1)
stats["coin_area_mean"] = round(float(np.mean([f["area"] for f in coin_features])), 1)

# 取最大的一枚做多边形近似演示（Douglas-Peucker）
big = max(coin_cnts, key=cv2.contourArea)
bp = cv2.arcLength(big, True)
apx1 = cv2.approxPolyDP(big, 0.01 * bp, True)
apx2 = cv2.approxPolyDP(big, 0.04 * bp, True)
stats["coin_poly_verts_eps001"] = len(apx1)
stats["coin_poly_verts_eps004"] = len(apx2)
stats["coin_big_area"] = round(float(cv2.contourArea(big)), 1)
stats["coin_big_perimeter"] = round(float(bp), 1)
stats["coin_big_circularity"] = round(float(circularity(big)), 4)

# 画所有候选 + 仅标出 7 枚硬币
coins_rgb = cv2.cvtColor(coins, cv2.COLOR_GRAY2RGB)
cv2.drawContours(coins_rgb, cands, -1, (255, 170, 0), 1)       # 候选(橙)
cv2.drawContours(coins_rgb, coin_cnts, -1, (0, 200, 0), 2)     # 7 枚硬币(绿)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(coins, cmap="gray"); axes[0].set_title("coins original"); axes[0].axis("off")
axes[1].imshow(coins_rgb)
axes[1].set_title(f"{len(coin_cnts)} coins (from {len(cands)} round candidates / {stats['contour_raw_count']} raw)")
axes[1].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_contours.png")); plt.close(fig)

# ---------- 6. 模板匹配（coins：用一枚硬币当模板，多尺度定位同类硬币） ----------
# 取面积居中的一枚当模板（避免取最大/最小导致尺度偏差）
mid = coin_cnts[len(coin_cnts) // 2]
x, y, w, h = cv2.boundingRect(mid)
pad = 6
x0, y0 = max(0, x - pad), max(0, y - pad)
x1, y1 = min(coins.shape[1], x + w + pad), min(coins.shape[0], y + h + pad)
tmpl = coins[y0:y1, x0:x1].astype(np.float32)
stats["template_shape"] = [int(tmpl.shape[1]), int(tmpl.shape[0])]

# 多尺度：在金字塔各层做匹配，找最佳尺度
best_score = -1.0
best_level = -1
for k, p in enumerate(pyr):
    scale = 1.0 / (2 ** k)
    tw = max(1, int(tmpl.shape[1] * scale))
    th = max(1, int(tmpl.shape[0] * scale))
    if p.shape[0] <= th or p.shape[1] <= tw:
        continue
    t_resized = cv2.resize(tmpl, (tw, th))
    res = cv2.matchTemplate(p, t_resized, cv2.TM_CCOEFF_NORMED)
    _, mx, _, _ = cv2.minMaxLoc(res)
    if mx > best_score:
        best_score = mx
        best_level = k
stats["template_best_pyramid_level"] = best_level
stats["template_best_pyramid_score"] = round(float(best_score), 4)

# 原尺度精细匹配：定位与模板相似的硬币（同类面额），逐峰 NMS 取点
res_full = cv2.matchTemplate(coins.astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED)
stats["template_best_score_fullscale"] = round(float(res_full.max()), 4)
thr = 0.8
peaks = []
work = res_full.copy()
for _ in range(50):
    _, mx, _, loc = cv2.minMaxLoc(work)
    if mx < thr:
        break
    px, py = loc
    peaks.append((px, py, round(float(mx), 3)))
    cv2.rectangle(work, (px - 12, py - 12), (px + 12, py + 12), 0.0, -1)
stats["template_score_thr"] = thr
stats["template_matches"] = len(peaks)
stats["template_match_scores"] = [s for _, _, s in peaks]

# 画模板 + 相关图 + 命中
coins_rgb2 = cv2.cvtColor(coins, cv2.COLOR_GRAY2RGB)
tw, th = tmpl.shape[1], tmpl.shape[0]
for (px, py, s) in peaks:
    cv2.rectangle(coins_rgb2, (px, py), (px + tw, py + th), (200, 0, 0), 2)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(tmpl, cmap="gray"); axes[0].set_title(f"template ({tw}x{th})"); axes[0].axis("off")
axes[1].imshow(res_full, cmap="jet"); axes[1].set_title("match correlation map"); axes[1].axis("off")
axes[2].imshow(coins_rgb2); axes[2].set_title(f"matched: {len(peaks)} same-denomination"); axes[2].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_template.png")); plt.close(fig)

# ---------- 7. 边缘检测效果横向对比（柱状图） ----------
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
labels = ["Sobel", "Scharr", "Laplacian", "Canny(hand-made)"]
ratios = [sobel_ratio, scharr_ratio, lap_ratio, stats["canny_final_ratio"]]
bars = ax.bar(labels, [r * 100 for r in ratios], color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
for b, r in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{r*100:.1f}%", ha="center", fontsize=11)
ax.set_ylabel("edge pixel ratio (%)")
ax.set_title("edge pixel ratio across operators (camera 512x512)")
ax.set_ylim(0, max([r * 100 for r in ratios]) * 1.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_edge_compare.png")); plt.close(fig)

with open(os.path.join(ROOT, "..", "stats.json"), "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print("DONE", json.dumps(stats, ensure_ascii=False))
