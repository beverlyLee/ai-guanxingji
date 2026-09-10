"""小红书竖版配图合成（3:4, 1080x1440）。

原则：图上所有数字都来自 M8/stats.json，底图一律复用 figures/ 里的真实
matplotlib 输出，绝不用 AI 重绘图表（会编造数字）。
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
OUT = ROOT / "发布物料" / "配图"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1440
S = 2  # 超采样倍数
BG = (255, 253, 248)
CARD = (255, 255, 255)
CORAL = (232, 101, 90)
INK = (58, 58, 58)
GRAY = (150, 145, 140)
MINT = (168, 216, 200)

FONT_M = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_L = "/System/Library/Fonts/STHeiti Light.ttc"


def font(path, size):
    return ImageFont.truetype(path, size * S)


def fit(im, box_w, box_h):
    """等比缩放到框内（宽优先，超高再按高缩）。"""
    r = min(box_w / im.width, box_h / im.height)
    return im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)


def trim_white(im, tol=247, keep=6):
    """裁掉 matplotlib 导出时四周的纯白边，让图在卡片里更饱满。"""
    import numpy as np
    g = np.asarray(im.convert("L"))
    m = g < tol
    if not m.any():
        return im
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    x0 = max(0, cols[0] - keep)
    y0 = max(0, rows[0] - keep)
    x1 = min(im.width, cols[-1] + 1 + keep)
    y1 = min(im.height, rows[-1] + 1 + keep)
    return im.crop((x0, y0, x1, y1))


def wrap(draw, text, fnt, max_w):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=fnt) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def card(idx, title, subs, note, out_name):
    """subs: [(图片文件名, 该图在可用高度中的占比权重)]，按顺序上下排列。"""
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)

    f_title = font(FONT_M, 50)
    f_note = font(FONT_L, 27)
    f_src = font(FONT_L, 21)

    pad = 64 * S
    y = 66 * S

    # 编号圆点
    r = 30 * S
    d.ellipse([pad, y, pad + 2 * r, y + 2 * r], fill=CORAL)
    f_num = font(FONT_M, 30)
    tw = d.textlength(str(idx), font=f_num)
    d.text((pad + r - tw / 2, y + r - 20 * S), str(idx), font=f_num, fill=(255, 255, 255))

    # 标题
    tx = pad + 2 * r + 22 * S
    tlines = wrap(d, title, f_title, W * S - tx - pad)
    ty = y + 2 * S
    for ln in tlines:
        d.text((tx, ty), ln, font=f_title, fill=INK)
        ty += 64 * S
    y = max(ty, y + 2 * r) + 22 * S

    # 顶部细线
    d.line([pad, y, W * S - pad, y], fill=CORAL, width=int(2.5 * S))
    y += 26 * S

    # 图区
    note_h = (len(note) * 44 + 30) * S
    boxes_top = y
    boxes_h = H * S - boxes_top - note_h - 88 * S
    gap = 22 * S
    total_w = sum(w for _, w, _ in subs)
    avail_h = boxes_h - gap * (len(subs) - 1)
    y_end = boxes_top

    for name, weight, crop in subs:
        hh = int(avail_h * weight / total_w)
        src = Image.open(FIG / name).convert("RGB")
        if crop:
            src = src.crop(crop)
        src = trim_white(src)
        scaled = fit(src, (W - 2 * pad) * S, hh)
        # 白底圆角承托
        cx0 = (W * S - scaled.width) // 2
        cy0 = boxes_top + (hh - scaled.height) // 2
        pad_in = 10 * S
        d.rounded_rectangle(
            [cx0 - pad_in, cy0 - pad_in, cx0 + scaled.width + pad_in, cy0 + scaled.height + pad_in],
            radius=18 * S, fill=CARD, outline=(236, 232, 226), width=int(1.5 * S),
        )
        img.paste(scaled, (cx0, cy0))
        boxes_top += hh + gap
    y_end = boxes_top - gap

    # 底部结论（贴近图区，不让画布底部留空）
    y = min(y_end + 26 * S, H * S - note_h - 62 * S)
    d.rounded_rectangle([pad, y, W * S - pad, H * S - 62 * S], radius=16 * S, fill=(255, 240, 236))
    ty = y + 16 * S
    for ln in note:
        d.text((pad + 22 * S, ty), ln, font=f_note, fill=(150, 60, 52))
        ty += 44 * S

    d.text((pad, H * S - 46 * S), "数据来源 UCI Iris（150 条）· 实测于 M8/experiment.py · 可复现",
           font=f_src, fill=GRAY)

    img = img.resize((W, H), Image.LANCZOS)
    img.save(OUT / out_name, quality=95)
    print("saved", out_name, img.size)


def main():
    # 卡 2：softmax 三概率
    card(2, "AI 给的不是答案，是三个概率",
         [("fig2_softmax_proba.png", 1.0, None)],
         ["这朵边界上的花，模型给的是 0 / 0.456 / 0.544",
          "top1 和 top2 只差 0.0884，它自己也没底"],
         "card2_three_probs.png")

    # 卡 3：损失下降
    card(3, "两千步，损失从 1.0986 掉到 0.0407",
         [("fig3_train_loss.png", 1.0, None)],
         ["1.0986 就是三分类纯瞎猜时的损失（ln3）",
          "两千步梯度下降之后，它开始有判断了"],
         "card3_loss.png")

    # 卡 4：梯度校验
    card(4, "自己推的梯度，到底对不对",
         [("fig4_grad_check.png", 1.0, None)],
         ["解析梯度 vs 数值梯度，最大偏差 1.46e-10",
          "手写实现与 sklearn 测试准确率都是 0.9111"],
         "card4_grad_check.png")

    # 卡 5：线性 vs 弯边界（上下两图，宽高比接近，排布整齐）
    card(5, "分类线能画弯，但不是越弯越好",
         [("fig5_linear_boundary.png", 1 / 1.231, None),
          ("fig6_nonlinear_boundary.png", 1 / 1.167, (1680, 0, 3360, 1440))],
         ["花瓣特征：直线 0.9111，三次多项式 0.9333",
          "花萼特征：直线 0.7556，弯了反而掉到 0.6889（过拟合）"],
         "card5_boundary.png")

    # 卡 6：错在哪
    card(6, "45 个测试样本，错的那 4 个在哪",
         [("fig1_iris_overlap.png", 1 / 1.333, None), ("fig7_confusion.png", 1 / 1.12, None)],
         ["对 41 个、错 4 个，错的全挤在两个长得像的物种中间",
          "山鸢尾一个没错，它跟另外两类分得很开"],
         "card6_errors.png")


if __name__ == "__main__":
    main()
