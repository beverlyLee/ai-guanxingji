import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

DIR = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/assets/ai-stargazing-h1-illustrations"
FONT = "/Users/liboyang/.workbuddy/skills/xiaotao-illustrations/assets/fonts/LXGWWenKai-Regular.ttf"

# --- Step 1: rename generated base to a working name ---
base_src = os.path.join(DIR, "_角色锁定__A_chibi_style_character_2026-08-24T10-51-56.png")
base_work = os.path.join(DIR, "grad-iter_base.png")
if os.path.exists(base_src):
    os.replace(base_src, base_work)
else:
    # fallback if filename differs
    raise SystemExit(f"base not found: {base_src}")

# --- Step 2: programmatic label overlay (NEVER model-drawn) ---
COLOR = (111, 53, 49)   # #6F3531 deep warm brown-red
FS = 38
font = ImageFont.truetype(FONT, FS)

labels = [
    ("第1步", (150, 165)),
    ("第2步", (440, 285)),
    ("第3步", (730, 405)),
    ("谷底",  (970, 525)),
]

img = Image.open(base_work).convert("RGBA")
for text, xy in labels:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text(xy, text, font=font, fill=COLOR + (255,))
    img = Image.alpha_composite(img, layer)

out = os.path.join(DIR, "grad-iter.png")
img.convert("RGB").save(out)
print("wrote", out, os.path.getsize(out), "bytes")

# --- Step 3: mandatory corner self-check (per team-lead) ---
for p in ["grad-tangent.png", "grad-iter.png"]:
    path = os.path.join(DIR, p)
    a = np.array(Image.open(path).convert("RGB"))
    H, W, _ = a.shape
    white = (a.mean(2) > 245).mean()
    cw, ch = W // 6, H // 6
    def c(x0, y0, x1, y1):
        return (a[y0:y1, x0:x1].mean(2) < 240).mean()
    print(p, f"white={white:.3f}",
          f"TL={c(0,0,cw,ch):.3f} TR={c(W-cw,0,W,ch):.3f}",
          f"BL={c(0,H-ch,cw,H):.3f} BR={c(W-cw,H-ch,W,H):.3f}")
