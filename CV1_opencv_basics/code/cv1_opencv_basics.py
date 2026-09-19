"""
CV1 OpenCV 基础 — 用 scikit-image 公共测试图做真实实验，产出 figures + stats.json
运行环境：受管 python venv（需 opencv-python-headless + scikit-image + numpy + matplotlib + scipy）
所有数字真实可复现，写入 ../stats.json
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

# ---------- 1. 图像即矩阵 ----------
page = skd.page()        # 灰度扫描文档页 (H, W) uint8
camera = skd.camera()   # 灰度 512x512
coins = skd.coins()     # 灰度 303x384
astro = skd.astronaut() # RGB 512x512x3

stats["shapes"] = {
    "page": list(page.shape), "camera": list(camera.shape),
    "coins": list(coins.shape), "astronaut": list(astro.shape),
}
stats["page_dtype"] = str(page.dtype)
stats["page_range"] = [int(page.min()), int(page.max())]
stats["page_mean_std"] = [round(float(page.mean()), 2), round(float(page.std()), 2)]

# fig1: page 灰度 + 直方图
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(page, cmap="gray"); ax[0].set_title("page grayscale (scanned doc w/ shadow & noise)"); ax[0].axis("off")
ax[1].hist(page.ravel(), bins=64, color="#444444"); ax[1].set_title("page pixel histogram")
ax[1].set_xlabel("pixel value 0-255"); ax[1].set_ylabel("count")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_page_hist.png")); plt.close(fig)

# BGR 陷阱：cv2 写盘再读回，演示通道顺序
cv2.imwrite(os.path.join(DATA, "astro_bgr.png"), cv2.cvtColor(astro, cv2.COLOR_RGB2BGR))
bgr = cv2.imread(os.path.join(DATA, "astro_bgr.png"))
stats["bgr_channel_swapped"] = bool(bgr[..., 0].sum() != astro[..., 0].sum())

# ---------- 2. ROI 切片 ----------
y0, y1, x0, x1 = 30, 130, 30, 130
coin_roi = coins[y0:y1, x0:x1]
stats["roi_slice"] = [y0, y1, x0, x1]
stats["roi_shape"] = list(coin_roi.shape)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(coins, cmap="gray")
ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="r", fill=False))
ax[0].set_title("coins + ROI box"); ax[0].axis("off")
ax[1].imshow(coin_roi, cmap="gray"); ax[1].set_title("ROI slice"); ax[1].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_roi.png")); plt.close(fig)

# ---------- 3. 边界填充 ----------
img_b = coins[100:140, 100:140].astype(np.uint8)
modes = [("CONSTANT", cv2.BORDER_CONSTANT), ("REPLICATE", cv2.BORDER_REPLICATE),
         ("REFLECT", cv2.BORDER_REFLECT), ("WRAP", cv2.BORDER_WRAP)]
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
axes[0].imshow(img_b, cmap="gray"); axes[0].set_title("orig"); axes[0].axis("off")
for i, (name, mode) in enumerate(modes, start=1):
    p = cv2.copyMakeBorder(img_b, 10, 10, 10, 10, mode, value=0)
    axes[i].imshow(p, cmap="gray"); axes[i].set_title(name); axes[i].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_border.png")); plt.close(fig)
stats["border_modes"] = [m[0] for m in modes]

# ---------- 4. 阈值 ----------
_, gth = cv2.threshold(coins, 127, 255, cv2.THRESH_BINARY)
oth, otsu = cv2.threshold(coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
stats["threshold_global_val"] = 127
stats["threshold_otsu_val"] = int(oth)
stats["coins_fg_ratio_global"] = round(float((gth > 0).mean()), 4)
stats["coins_fg_ratio_otsu"] = round(float((otsu > 0).mean()), 4)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(ax, [coins, gth, otsu], ["coins orig", "global thr 127", "OTSU auto"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_threshold.png")); plt.close(fig)

othp, otsu_p = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
stats["page_otsu_val"] = int(othp)

# ---------- 5. 平滑去噪 ----------
# 稀疏椒盐噪点（各 6%），才能体现中值滤波相对均值/高斯的优势
rng = np.random.default_rng(42)
salt = rng.random(page.shape) < 0.06
pep = rng.random(page.shape) < 0.06
page_n = page.astype(np.int16)
page_n[salt] = 255
page_n[pep] = 0
page_n = np.clip(page_n, 0, 255).astype(np.uint8)
mean_b = cv2.blur(page_n, (5, 5))
gauss_b = cv2.GaussianBlur(page_n, (5, 5), 0)
median_b = cv2.medianBlur(page_n, 5)
stats["denoise_noise_std_raw"] = round(float(page_n.std()), 2)
stats["denoise_std_mean"] = round(float(mean_b.std()), 2)
stats["denoise_std_gauss"] = round(float(gauss_b.std()), 2)
stats["denoise_std_median"] = round(float(median_b.std()), 2)
stats["denoise_clean_page_std"] = round(float(page.std()), 2)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for a, im, t in zip(axes, [page_n, mean_b, gauss_b, median_b], ["noisy page", "mean blur", "gaussian blur", "median blur"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_smoothing.png")); plt.close(fig)

# ---------- 6. 形态学 ----------
bin_coins = (otsu > 0).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(bin_coins, kernel, iterations=1)
dilated = cv2.dilate(bin_coins, kernel, iterations=1)
opened = cv2.morphologyEx(bin_coins, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(bin_coins, cv2.MORPH_CLOSE, kernel)


def ncomp(b):
    lab, n = ndimage.label(b)
    return int(n)


stats["morph_components_raw"] = ncomp(bin_coins)
stats["morph_components_open"] = ncomp(opened)
stats["morph_components_close"] = ncomp(closed)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for a, im, t in zip(axes.ravel(), [eroded, dilated, opened, closed],
                    ["erode", "dilate", "open (rm small noise)", "close (fill holes)"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_morphology.png")); plt.close(fig)

# ---------- 7. 梯度 Sobel / Laplacian ----------
gx = cv2.Sobel(camera, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(camera, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(gx ** 2 + gy ** 2)
lap = cv2.Laplacian(camera, cv2.CV_64F, ksize=3)
stats["sobel_edge_pixel_ratio"] = round(float((mag > 0.15 * mag.max()).mean()), 4)
stats["sobel_mag_max"] = round(float(mag.max()), 2)
stats["laplacian_max"] = round(float(np.abs(lap).max()), 2)


def shownorm(x):
    x = np.abs(x)
    return x / x.max()


fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for a, im, t in zip(axes, [shownorm(gx), shownorm(gy), shownorm(mag), shownorm(lap)],
                    ["Sobel-x", "Sobel-y", "gradient magnitude", "Laplacian"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_gradient.png")); plt.close(fig)

with open(os.path.join(ROOT, "..", "stats.json"), "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print("DONE", json.dumps(stats, ensure_ascii=False))
