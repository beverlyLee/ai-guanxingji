from PIL import Image, ImageDraw, ImageFont

FONT_CN = "/Users/liboyang/.workbuddy/skills/xiaotao-illustrations/assets/fonts/LXGWWenKai-Regular.ttf"
IN = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/assets/ai-stargazing-h1-illustrations/grad-iter-base.png"
OUT = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/assets/ai-stargazing-h1-illustrations/grad-iter.png"

INK = (111, 53, 49)

img = Image.open(IN).convert("RGBA")
W, H = img.size  # 1280, 720

font_big = ImageFont.truetype(FONT_CN, 40)   # 第1/2/3 步
font_label = ImageFont.truetype(FONT_CN, 44) # 谷底


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(layer, text, center_xy, font, angle=0):
    tmp = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    w, h = text_size(d, text, font)
    xy = (int(center_xy[0] - w / 2), int(center_xy[1] - h / 2))
    d.text(xy, text, font=font, fill=INK + (255,))
    if angle:
        tmp = tmp.rotate(angle, expand=False, center=center_xy)
    return Image.alpha_composite(layer, tmp)


layer = Image.new("RGBA", img.size, (0, 0, 0, 0))

# Labels placed by visually inspecting grad-iter-base.png:
# text center (x, y) in pixels
labels = [
    ("第1步", (960, 290), font_big, 0),   # on reserved empty surface of platform 1 (right side, clear of foot)
    ("第2步", (800, 360), font_big, 0),   # on exposed upper-right area of platform 2, clear of XiaoTao's body
    ("第3步", (545, 525), font_big, 0),   # on reserved empty surface of platform 3
    ("谷底",  (300, 540), font_label, 0), # right of U-valley bottom
]

for text, center, font, angle in labels:
    layer = draw_centered(layer, text, center, font, angle)
    print(f"Placed '{text}' at center {center}")

out = Image.alpha_composite(img, layer)
out.convert("RGB").save(OUT, "PNG", optimize=True)
print(f"Saved: {OUT}")
