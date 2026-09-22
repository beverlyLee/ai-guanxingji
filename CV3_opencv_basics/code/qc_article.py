# -*- coding: utf-8 -*-
"""CV3 文章 QC：黑名单词 / 破折号 / 粗体 / 数字一致 / 字数 / 数学占比。
用法：受管 python 跑此脚本，依赖本目录的 stats.json 与 ../CV3_手机修图里的一键去雾到底动了什么.md
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "CV3_手机修图里的一键去雾到底动了什么.md"
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

# 4. 字数（粗略中文字数）
cn = len(re.findall(r"[一-鿿]", text))
checks.append(("中文字数", f"{cn} (目标约10000)"))

# 5. 代码/公式块数
math_blocks = len(re.findall(r"```", text)) // 2
checks.append(("代码/公式块数", f"{math_blocks}"))

# 6. 数字一致性：抽取正文里关键数字，和 stats 比对（以正文展示字符串为准）
num_checks = {
    "像素总数 262144": "262144" in text,
    "最频灰度级样本数 4957": "4957" in text,
    "最频占比 1.89": "1.89" in text,
    "原图标准差 0.2888": "0.2888" in text,
    "原图熵 7.2317": "7.2317" in text,
    "退化雾图 t=0.35": "0.35" in text,
    "雾图标准差 0.1011": "0.1011" in text,
    "雾图标准差下降 65%": "65%" in text,
    "雾图均衡化后标准差 0.1821": "0.1821" in text,
    "雾图均衡化增益 1.8": "1.8" in text,
    "雾图均衡化熵 5.5527": "5.5527" in text,
    "傅里叶 DC 幅度 132676.5": "132676.5" in text,
    "低频能量占比(半径30) 98.46": "98.46" in text,
    "高频能量占比(半径60) 0.78": "0.78" in text,
    "低通能量保留 97.28": "97.28" in text,
    "低通标准差 0.2723": "0.2723" in text,
    "低通标准差下降 5.71": "5.71" in text,
    "高通能量保留 1.58": "1.58" in text,
    "原图边缘能量 0.03422": "0.03422" in text,
    "低通边缘能量 0.00975": "0.00975" in text,
    "高通边缘能量 0.01797": "0.01797" in text,
    "低通边缘能量比 28.5%": "28.5" in text,
    "高通边缘能量比 52.5%": "52.5" in text,
    "原图均衡化标准差 0.2886": "0.2886" in text,
    "原图最频灰度CDF 0.1715": "0.1715" in text,
    "低频能量占比(半径10) 96.84": "96.84" in text,
    "总能量 2.333×10": "2.333×10" in text,
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
    print(f"  {name:<30} {status}")
print("=" * 48)
if fails:
    print(f"QC FAIL: {len(fails)} 项未过")
    for tag, info in fails:
        print(f"  - {tag}: {info}")
    sys.exit(1)
else:
    print("QC PASS：全部通过")
