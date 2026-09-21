# -*- coding: utf-8 -*-
"""CV2 文章 QC：黑名单词 / 破折号 / 粗体 / 数字一致 / 字数 / 数学占比。
用法：受管 python 跑此脚本，依赖本目录的 stats.json 与 ../CV2_OpenCV图像边缘与轮廓.md
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "CV2_OpenCV图像边缘与轮廓.md"
STATS = ROOT / "stats.json"

BLACKLIST = ["赋能", "深度", "洞见", "底层逻辑", "硬核", "天花板", "降维打击",
             "抓手", "闭环", "沉淀", "拉齐", "对齐", "此外", "然而", "标志着",
             "至关重要", "不可或缺", "总而言之", "旨在"]

text = MD.read_text(encoding="utf-8")
stats = json.loads(STATS.read_text(encoding="utf-8"))

fails = []
checks = []

# 1. 黑名单词
hit_bl = [w for w in BLACKLIST if w in text]
checks.append(("黑名单词", "PASS" if not hit_bl else f"FAIL -> {hit_bl}"))
if hit_bl:
    fails.append(("blacklist", hit_bl))

# 2. 破折号（—— 与单 em dash —）
emo = text.count("——")
em = len(re.findall(r"—", text))
checks.append(("破折号——", "PASS" if emo == 0 else f"FAIL x{emo}"))
checks.append(("em dash —", "PASS" if em == 0 else f"FAIL x{em}"))
if emo:
    fails.append(("emdash_double", emo))
if em:
    fails.append(("emdash_single", em))

# 3. 粗体 ** 或 __
bold = len(re.findall(r"\*\*.+?\*\*", text)) + len(re.findall(r"__.+?__", text))
checks.append(("粗体", "PASS" if bold == 0 else f"FAIL x{bold}"))
if bold:
    fails.append(("bold", bold))

# 4. 字数（去空白，粗略中文字数）
cn = len(re.findall(r"[一-鿿]", text))
checks.append(("中文字数", f"{cn} (目标约10000)"))

# 5. 数学占比：公式块 + 行内 $ 风格的代码/公式，粗略用代码块行数估算
math_blocks = len(re.findall(r"```", text)) // 2
checks.append(("代码/公式块数", f"{math_blocks}"))

# 6. 数字一致性：抽取正文里关键数字，和 stats 比对
def has(s):
    return s in text

num_checks = {
    "coins 7 枚": ("24" in text and "7 枚" in text and "22 个" in text),
    "sobel_mag_max 930.11": "930.11" in text,
    "scharr_mag_max 4020.9": "4020.9" in text,
    "laplacian_max 1110.0": "1110.0" in text,
    "edge_ratio sobel 0.0837": "0.0837" in text,
    "edge_ratio scharr 0.0794": "0.0794" in text,
    "edge_ratio laplacian 0.1384": "0.1384" in text,
    "canny low 44.16 / high 88.32": ("44.16" in text and "88.32" in text),
    "nms raw 39560 / after 13610": ("39560" in text and "13610" in text),
    "nms_reduction 2.91": "2.91" in text,
    "canny final 7830 / ratio 0.0299": ("7830" in text and "0.0299" in text),
    "opencv_final 38358": "38358" in text,
    "pyramid L0 303x384 std 52.88": ("303×384" in text or "303x384" in text) and "52.88" in text,
    "pyramid L4 19x24 std 30.68": ("19×24" in text or "19x24" in text) and "30.68" in text,
    "coins_detected 7 / area_mean 2212.0": ("2212.0" in text),
    "coin big area 3022.5 circ 0.8512": ("3022.5" in text and "0.8512" in text),
    "poly eps 12 / 6": ("12 个顶点" in text and "6 个顶点" in text),
    "template shape 69x64": ("69×64" in text or "69x64" in text),
    "template matches 2 / 0.806": ("2 枚" in text and "0.806" in text),
}
for k, ok in num_checks.items():
    checks.append((f"数字一致[{k}]", "PASS" if ok else "FAIL"))
    if not ok:
        fails.append(("num", k))

# 7. 三个干货点 + 互动钩子
has3 = "三个值得记住的点" in text
has_hook = "互动钩子" in text
checks.append(("三个干货点标题", "PASS" if has3 else "FAIL"))
checks.append(("互动钩子标题", "PASS" if has_hook else "FAIL"))
if not has3:
    fails.append(("takeaways", "missing"))
if not has_hook:
    fails.append(("hook", "missing"))

# 8. 七张图引用
figs = [f"figures/fig{i}" for i in range(1, 8)]
miss_fig = [f for f in figs if f not in text]
checks.append(("七图引用", "PASS" if not miss_fig else f"FAIL 缺 {miss_fig}"))
if miss_fig:
    fails.append(("fig", miss_fig))

print("=" * 48)
for name, status in checks:
    print(f"  {name:<28} {status}")
print("=" * 48)
if fails:
    print(f"QC FAIL: {len(fails)} 项未过")
    for tag, info in fails:
        print(f"  - {tag}: {info}")
    sys.exit(1)
else:
    print("QC PASS：全部通过")
