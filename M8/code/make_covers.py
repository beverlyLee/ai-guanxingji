"""M8 两张封面的裁切与文字叠加。

- 小红书封面：3:4（1080x1440），叠中文标题（程序绘制，避免 AI 出乱字）
- 公众号封面：1920x817（2.35:1），纯视觉不加字
底图由 ImageGen 生成，存于 发布物料/配图/_raw/
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "发布物料" / "配图" / "_raw"
OUT = ROOT / "发布物料" / "配图"

FONT_M = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_L = "/System/Library/Fonts/STHeiti Light.ttc"
CORAL = (232, 101, 90)
INK = (43, 43, 43)
GRAY = (140, 134, 128)


def cover(im, w, h, y_bias=0.0):
    """按 cover 方式缩放并居中裁切；y_bias>0 表示裁切窗口向下偏移（多保留顶部）。"""
    r = max(w / im.width, h / im.height)
    im2 = im.resize((int(im.width * r + 0.5), int(im.height * r + 0.5)), Image.LANCZOS)
    x = (im2.width - w) // 2
    y = (im2.height - h) // 2 + int(y_bias * (im2.height - h))
    y = max(0, min(y, im2.height - h))
    return im2.crop((x, y, x + w, y + h))


def make_xhs():
    S = 2
    W, H = 1080, 1440
    base = cover(Image.open(RAW / "xhs_cover_raw.png").convert("RGB"), W * S, H * S, y_bias=-0.10)
    d = ImageDraw.Draw(base, "RGBA")

    pad = 78 * S
    # 标题承托（柔白块，压住网格线，保证文字可读）
    d.rounded_rectangle([pad - 26 * S, 92 * S, W * S - pad + 26 * S, 470 * S],
                        radius=26 * S, fill=(255, 253, 248, 205))

    f_tag = ImageFont.truetype(FONT_M, 30 * S)
    f_h1 = ImageFont.truetype(FONT_M, 80 * S)
    f_sub = ImageFont.truetype(FONT_L, 36 * S)

    # 顶部小标签
    tag = "机器学习 · 多分类逻辑回归"
    tw = d.textlength(tag, font=f_tag)
    d.rounded_rectangle([pad, 128 * S, pad + tw + 40 * S, 128 * S + 58 * S],
                        radius=29 * S, fill=CORAL)
    d.text((pad + 20 * S, 128 * S + 12 * S), tag, font=f_tag, fill=(255, 255, 255))

    # 主标题两行
    y = 214 * S
    for i, ln in enumerate(["AI 说这蘑菇能吃", "它自己有几成把握？"]):
        d.text((pad, y), ln, font=f_h1, fill=INK if i == 0 else CORAL)
        y += 96 * S

    # 副标
    d.text((pad, y + 10 * S), "用鸢尾花跑一遍多分类逻辑回归", font=f_sub, fill=GRAY)

    base = base.resize((W, H), Image.LANCZOS)
    base.save(OUT / "cover_xiaohongshu.png", quality=95)
    print("saved cover_xiaohongshu.png", base.size)


def make_mp():
    base = cover(Image.open(RAW / "mp_cover_raw.png").convert("RGB"), 1920, 817, y_bias=-0.80)
    base.save(OUT / "cover_gongzhonghao_1920x817.png", quality=95)
    print("saved cover_gongzhonghao_1920x817.png", base.size)


if __name__ == "__main__":
    make_xhs()
    make_mp()
