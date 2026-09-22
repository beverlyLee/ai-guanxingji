"""CV3 实验脚本：直方图 / 均衡化 / 傅里叶 / 频域低通高通滤波。

所有正文引用的数字都来自本脚本的输出 stats.json，保证可复现、数据可溯源。
运行：python code/experiment.py  （在 CV3_opencv_basics 目录下）
依赖：numpy / matplotlib / scikit-image / scipy
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from skimage import data, img_as_float, exposure
from skimage.filters import sobel
from skimage.color import rgb2gray

# 注册系统中文矢量字体（macOS 自带，保证中文标题可渲染；换机请改路径）
_CJK_FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
try:
    fm.fontManager.addfont(_CJK_FONT)
    _cjk_name = fm.FontProperties(fname=_CJK_FONT).get_name()
    plt.rcParams["font.family"] = _cjk_name
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
try:
    FIG.mkdir(parents=True)
except FileExistsError:
    pass
DATA = ROOT / "data"
try:
    DATA.mkdir(parents=True)
except FileExistsError:
    pass

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

stats = {}

# ----------------------------------------------------------------------------
# 0. 载入公开测试图（scikit-image 自带 camera，512x512 灰度，零敏感数据）
# ----------------------------------------------------------------------------
img_color = data.camera()
img = img_as_float(img_color) if img_color.ndim == 2 else img_as_float(rgb2gray(img_color))
H, W = img.shape
N = H * W
stats["img_h"] = int(H)
stats["img_w"] = int(W)
stats["img_pixels"] = int(N)

# ----------------------------------------------------------------------------
# 1. 直方图定义：灰度分布统计
# ----------------------------------------------------------------------------
hist, _ = np.histogram(img, bins=256, range=(0.0, 1.0))
mode_bin = int(np.argmax(hist))          # 最频灰度级（0-255）
mode_value = mode_bin / 255.0
mode_count = int(hist.max())
mean_gray = float(img.mean())
std_gray = float(img.std())
dark_frac = float((img < 0.2).mean())    # 暗像素占比
bright_frac = float((img > 0.8).mean())  # 亮像素占比

stats["hist_mode_bin"] = mode_bin
stats["hist_mode_value"] = round(mode_value, 4)
stats["hist_mode_count"] = mode_count
stats["hist_mode_count_pct"] = round(mode_count / N * 100, 2)
stats["orig_mean"] = round(mean_gray, 4)
stats["orig_std"] = round(std_gray, 4)
stats["orig_dark_frac"] = round(dark_frac * 100, 2)
stats["orig_bright_frac"] = round(bright_frac * 100, 2)


def entropy_of(image):
    h, _ = np.histogram(image, bins=256, range=(0.0, 1.0))
    p = h[h > 0] / h.sum()
    return float(-np.sum(p * np.log2(p)))


stats["orig_entropy"] = round(entropy_of(img), 4)

# ----------------------------------------------------------------------------
# 2. 模拟“雾图”以呼应标题的一键去雾：大气散射退化模型 I = A(1-t) + t*J
#    t 越小雾越浓，A 为大气光（偏亮）。退化后直方图被挤压到高亮区、对比度骤降。
# ----------------------------------------------------------------------------
A = 0.85     # 大气光
t = 0.35     # 透射率（小 => 雾浓）
hazy = np.clip(A * (1 - t) + t * img, 0.0, 1.0)
hazy_mean = float(hazy.mean())
hazy_std = float(hazy.std())
hazy_hist, _ = np.histogram(hazy, bins=256, range=(0.0, 1.0))
hazy_mode_bin = int(np.argmax(hazy_hist))
hazy_mode_value = hazy_mode_bin / 255.0
hazy_entropy = entropy_of(hazy)
stats["hazy_A"] = A
stats["hazy_t"] = t
stats["hazy_mean"] = round(hazy_mean, 4)
stats["hazy_std"] = round(hazy_std, 4)
stats["hazy_std_drop_pct"] = round((1 - hazy_std / std_gray) * 100, 2)
stats["hazy_mode_bin"] = hazy_mode_bin
stats["hazy_mode_value"] = round(hazy_mode_value, 4)
stats["hazy_entropy"] = round(hazy_entropy, 4)

# ----------------------------------------------------------------------------
# 3. 直方图均衡化（去雾核心）：手动累积分布变换，给出可溯源的数字
#    公式 s_k = round((L-1) * sum_{j=0..k} p_r(j)), L=256
# ----------------------------------------------------------------------------
L = 256
pdf = hist / N
cdf = np.cumsum(pdf)                       # cdf[k] = P(X <= k)
s = np.round((L - 1) * cdf).astype(int)    # 映射后的新灰度级
# 用查找表把原图映射到均衡图
lut = s.astype(float) / (L - 1)
img_eq = lut[(img * (L - 1)).astype(int)].reshape(img.shape)
hazy_eq = lut[(hazy * (L - 1)).astype(int)].reshape(hazy.shape)
# skimage 内置均衡化做交叉验证
img_eq_ref = exposure.equalize_hist(img)
hazy_eq_ref = exposure.equalize_hist(hazy)
stats["eq_max_abs_diff"] = round(float(np.max(np.abs(img_eq - img_eq_ref))), 5)

# CDF 关键灰度级处的数字（正文做成小表，读者可溯源）
cdf_table = []
for k in [0, 16, 32, 48, 64, 96, 128, 160, 192, 224, 255]:
    cdf_table.append({
        "gray": k,
        "gray_norm": round(k / 255.0, 4),
        "cdf": round(float(cdf[k]), 4),
        "s": int(s[k]),
    })
stats["cdf_table"] = cdf_table
stats["eq_first_nonzero_gray"] = int(np.argmax(cdf > 0))
stats["eq_cdf_at_mode"] = round(float(cdf[mode_bin]), 4)

eq_std = float(img_eq.std())
hazy_eq_std = float(hazy_eq.std())
stats["img_eq_std"] = round(eq_std, 4)
stats["hazy_eq_std"] = round(hazy_eq_std, 4)
stats["hazy_eq_std_gain"] = round(hazy_eq_std / hazy_std, 2)
stats["hazy_eq_entropy"] = round(entropy_of(hazy_eq), 4)

# ----------------------------------------------------------------------------
# 4. 傅里叶变换概述：幅度谱、低频在中心、能量分布
# ----------------------------------------------------------------------------
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
mag = np.abs(fshift)
mag_log = np.log1p(mag)
dc = float(mag[H // 2, W // 2])            # 直流分量（中心）
total_energy = float(np.sum(mag ** 2))
stats["fft_dc_magnitude"] = round(dc, 1)
stats["fft_total_energy"] = float(f"{total_energy:.3e}")
# 中心低频圆盘（半径 D）占能量比例
Y, X = np.ogrid[:H, :W]
cy, cx = H // 2, W // 2
for D in [10, 30, 60]:
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    low_energy = np.sum(mag[dist <= D] ** 2)
    stats[f"fft_low_energy_pct_D{D}"] = round(low_energy / total_energy * 100, 2)
# 高频（中心圆盘外）能量占比
dist60 = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
high_energy = np.sum(mag[dist60 > 60] ** 2)
stats["fft_high_energy_pct_D60"] = round(high_energy / total_energy * 100, 2)

# ----------------------------------------------------------------------------
# 5. 频域低通滤波（高斯低通）：保留低频去高频 => 模糊
# ----------------------------------------------------------------------------
sigma = 20.0
glp = np.exp(-(dist60 ** 2) / (2 * sigma ** 2))
lp_energy_keep = np.sum((mag * glp) ** 2) / total_energy * 100
img_lp = np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(fshift * glp))), 0, 1)
stats["lp_sigma"] = sigma
stats["lp_energy_keep_pct"] = round(lp_energy_keep, 2)
stats["lp_std"] = round(float(img_lp.std()), 4)
stats["lp_std_drop_pct"] = round((1 - img_lp.std() / std_gray) * 100, 2)

# ----------------------------------------------------------------------------
# 6. 频域高通滤波（高斯高通）：保留高频去低频 => 边缘/锐化
# ----------------------------------------------------------------------------
ghp = 1.0 - glp
img_hp = np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(fshift * ghp))), 0, 1)
# 高通会去掉直流，图像偏暗，用 high-emphasis 加回少量原图便于观察
img_hp_vis = np.clip(img_hp + 0.5 * img, 0, 1)
stats["hp_energy_keep_pct"] = round(np.sum((mag * ghp) ** 2) / total_energy * 100, 2)

# 边缘能量对比（Sobel）
def edge_energy(im):
    return float(np.mean(sobel(im)))

e_orig = edge_energy(img)
e_lp = edge_energy(img_lp)
e_hp = edge_energy(img_hp)
stats["edge_energy_orig"] = round(e_orig, 5)
stats["edge_energy_lp"] = round(e_lp, 5)
stats["edge_energy_hp"] = round(e_hp, 5)
stats["edge_energy_lp_ratio"] = round(e_lp / e_orig, 3)
stats["edge_energy_hp_ratio"] = round(e_hp / e_orig, 3)

# ----------------------------------------------------------------------------
# 7. 配图
# ----------------------------------------------------------------------------
# fig1: 原图 + 原图直方图（直方图定义）
fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.4))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("原图：scikit-image 自带 camera（512×512 灰度）", fontsize=11)
ax[0].axis("off")
ax[1].bar(range(256), hist, width=1.0, color="#3a6ea5")
ax[1].axvline(mode_bin, color="#c0392b", lw=1.2, ls="--",
              label=f"最频灰度级 {mode_bin}（占比 {mode_count/N*100:.1f}%）")
ax[1].set_title("灰度直方图：横轴是灰度级 0–255，纵轴是该灰度级的像素个数", fontsize=11)
ax[1].set_xlabel("灰度级"); ax[1].set_ylabel("像素数")
ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig1.png"); plt.close(fig)

# fig2: 傅里叶幅度谱（对数显示，低频在中心）
fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.2))
ax[0].imshow(mag_log, cmap="gray")
ax[0].set_title("傅里叶幅度谱（对数显示）", fontsize=11)
ax[0].axis("off")
# 叠加中心低频圆盘示意
circle = plt.Circle((cx, cy), 60, color="#e67e22", fill=False, lw=1.5, ls="--")
ax[0].add_patch(circle)
ax[1].imshow(glp, cmap="gray")
ax[1].set_title(f"高斯低通掩膜（σ={sigma:.0f}）：中心透亮=保留低频", fontsize=11)
ax[1].axis("off")
fig.tight_layout(); fig.savefig(FIG / "fig2.png"); plt.close(fig)

# fig3: 雾图 vs 均衡化去雾（2x2）
fig, ax = plt.subplots(2, 2, figsize=(8.4, 7.2))
ax[0, 0].imshow(hazy, cmap="gray"); ax[0, 0].set_title("退化雾图（t=0.35）", fontsize=10); ax[0, 0].axis("off")
hh, _ = np.histogram(hazy, bins=256, range=(0, 1))
ax[0, 1].bar(range(256), hh, width=1.0, color="#7f8c8d")
ax[0, 1].set_title(f"雾图直方图：挤压到高亮区，对比度骤降（σ={hazy_std:.3f}）", fontsize=9.5)
ax[1, 0].imshow(hazy_eq, cmap="gray"); ax[1, 0].set_title("直方图均衡化去雾后", fontsize=10); ax[1, 0].axis("off")
heh, _ = np.histogram(hazy_eq, bins=256, range=(0, 1))
ax[1, 1].bar(range(256), heh, width=1.0, color="#27ae60")
ax[1, 1].set_title(f"去雾后直方图：铺满 0–255，对比度恢复（σ={hazy_eq_std:.3f}）", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "fig3.png"); plt.close(fig)

# fig4: 均衡化 CDF 变换步骤（原直方图 / CDF 曲线 / 均衡直方图，3 联）
fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.4))
ax[0].bar(range(256), hist, width=1.0, color="#3a6ea5")
ax[0].set_title("① 原始直方图（像素集中，对比度低）", fontsize=10.5)
ax[1].plot(range(256), cdf * (L - 1), color="#c0392b", lw=1.6)
ax[1].set_title("② 累积分布函数 CDF：s = (L-1)·Σp_r(j)，单调拉伸", fontsize=10.5)
ax[1].set_xlabel("原灰度级 k"); ax[1].set_ylabel("映射灰度级 s")
eqh, _ = np.histogram(img_eq, bins=256, range=(0, 1))
ax[2].bar(range(256), eqh, width=1.0, color="#27ae60")
ax[2].set_title("③ 均衡后直方图：灰度铺开，细节被拉开", fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "fig4.png"); plt.close(fig)

# fig5: 低通滤波结果（模糊）
fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.2))
ax[0].imshow(glp, cmap="gray"); ax[0].set_title(f"低通掩膜（σ={sigma:.0f}）", fontsize=10.5); ax[0].axis("off")
ax[1].imshow(img_lp, cmap="gray")
ax[1].set_title(f"低通滤波后：边缘能量只剩原图 {e_lp/e_orig*100:.1f}%，画面变糊", fontsize=10.5)
ax[1].axis("off")
fig.tight_layout(); fig.savefig(FIG / "fig5.png"); plt.close(fig)

# fig6: 高通滤波结果（边缘）
fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.2))
ax[0].imshow(ghp, cmap="gray"); ax[0].set_title(f"高通掩膜（1-低通，σ={sigma:.0f}）", fontsize=10.5); ax[0].axis("off")
ax[1].imshow(img_hp_vis, cmap="gray")
ax[1].set_title(f"高通滤波后：边缘能量升到原图 {e_hp/e_orig*100:.1f}%，轮廓被点亮", fontsize=10.5)
ax[1].axis("off")
fig.tight_layout(); fig.savefig(FIG / "fig6.png"); plt.close(fig)

# fig7: 综合四联对比
fig, ax = plt.subplots(2, 2, figsize=(8.4, 7.2))
ax[0, 0].imshow(img, cmap="gray"); ax[0, 0].set_title("原图", fontsize=10.5); ax[0, 0].axis("off")
ax[0, 1].imshow(img_eq, cmap="gray"); ax[0, 1].set_title("均衡化（对比度拉伸）", fontsize=10.5); ax[0, 1].axis("off")
ax[1, 0].imshow(img_lp, cmap="gray"); ax[1, 0].set_title("低通（模糊）", fontsize=10.5); ax[1, 0].axis("off")
ax[1, 1].imshow(img_hp_vis, cmap="gray"); ax[1, 1].set_title("高通（边缘/锐化）", fontsize=10.5); ax[1, 1].axis("off")
fig.tight_layout(); fig.savefig(FIG / "fig7.png"); plt.close(fig)

# ----------------------------------------------------------------------------
# 写出 stats.json（位于篇根目录，供 qc_article.py 比对）
# ----------------------------------------------------------------------------
with open(ROOT / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("=== CV3 实验完成，关键数字 ===")
for k in ["img_h", "img_w", "img_pixels", "hist_mode_bin", "hist_mode_count_pct",
          "orig_std", "orig_entropy", "hazy_t", "hazy_std", "hazy_std_drop_pct",
          "hazy_eq_std_gain", "hazy_eq_entropy", "fft_dc_magnitude",
          "fft_low_energy_pct_D30", "fft_high_energy_pct_D60",
          "lp_energy_keep_pct", "lp_std_drop_pct",
          "hp_energy_keep_pct", "edge_energy_lp_ratio", "edge_energy_hp_ratio"]:
    print(f"  {k} = {stats[k]}")
print("figures:", sorted(p.name for p in FIG.glob('fig*.png')))
