"""
M13 文章 QC：八项自检。全部通过才返回 0。
  1 字数（中文字符）不超过上限
  2 黑名单词 0
  3 破折号 0
  4 粗体 0
  5 数学占比不超过 20%
  6 关键结果数字与 stats.json 一致（带舍入容差）
  7 标题数、标题格式
  8 图片引用数且文件存在
附加：第一人称出现、末尾互动钩子。
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 正文学术稿：排除选题方案 / 规划类文件
cands = [p for p in ROOT.glob("M13_*.md")
         if not any(k in p.name for k in ("选题", "规划", "方案"))]
ART = max(cands, key=lambda p: p.stat().st_size)
stats_text = (ROOT / "stats.json").read_text(encoding="utf-8")
text = ART.read_text(encoding="utf-8")
print("正文文件：", ART.name)

fails, warns = [], []

# 1 字数
cn = len(re.findall(r"[\u4e00-\u9fff]", text))
print(f"[1] 中文 {cn} 字，总长 {len(text)}")
if cn > 8000:
    fails.append(f"字数超限 {cn}")

# 2 黑名单
BLACK = ["赋能", "洞见", "底层逻辑", "硬核", "天花板", "降维打击", "抓手", "闭环",
         "沉淀", "拉齐", "对齐", "此外", "然而", "标志着", "至关重要",
         "不可或缺", "总而言之", "旨在"]
hit = [w for w in BLACK if w in text]
print(f"[2] 黑名单命中 {hit}")
if hit:
    fails.append(f"黑名单 {hit}")

# 3 破折号
dash = text.count("——") + text.count("--")
print(f"[3] 破折号 {dash}")
if dash:
    fails.append(f"破折号 {dash}")

# 4 粗体
bold = text.count("**")
print(f"[4] 粗体标记 {bold}")
if bold:
    fails.append(f"粗体 {bold}")

# 5 数学占比（代码块 + 行内反引号）
code = sum(len(m) for m in re.findall(r"```.*?```", text, re.S))
inline = sum(len(m) for m in re.findall(r"`[^`\n]+`", text))
ratio = (code + inline) / len(text)
print(f"[5] 公式字符 {code + inline}，占比 {ratio:.1%}")
if ratio > 0.20:
    fails.append(f"数学占比 {ratio:.1%}")

# 6 关键结果数字：既能追溯到 stats.json，又确实用在了文中
NUM_KEYS = [
    (35064, 0), (84.2, 1), (78.8, 1), (24497, 0), (5260, 0),
    (0.9482, 4), (0.9241, 4), (22.74, 2), (27.53, 2),
    (0.0984, 4), (104.72, 2), (0.2587, 4), (86.02, 2), (57.09, 2),
    (0.9891, 4), (0.9179, 4), (0.9147, 4), (1.05, 2),
    (0.8877, 4), (0.8848, 4), (0.9006, 4), (0.9739, 4),
    (0.3502, 4), (0.1439, 4), (0.1476, 4),
    (0.137, 3), (0.2301, 4), (0.2031, 4),
    (0.2354, 4), (10.24, 2), (3.79, 2), (0.7003, 4), (0.8673, 4),
]
stats_nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", stats_text)]
not_used, not_trace = [], []
for v, d in NUM_KEYS:
    s = f"{v:.{d}f}"
    if s not in text:
        not_used.append(s)
    tol = 10 ** (-d - 1)
    if not any(abs(round(abs(n), d) - v) < tol for n in stats_nums):
        not_trace.append(s)
print(f"[6] 未用 {not_used}；不可追溯 {not_trace}")
if not_trace:
    fails.append(f"数字不可追溯 {not_trace}")
if not_used:
    warns.append(f"数字未用 {not_used}")

# 7 标题
heads = re.findall(r"^#\s+(.+)$", text, re.M)
h2 = re.findall(r"^##\s+(.+)$", text, re.M)
print(f"[7] 一级标题 {len(heads)}，二级 {len(h2)}")
if len(heads) != 1:
    fails.append(f"一级标题数 {len(heads)}")
if len(heads) + len(h2) < 3:
    fails.append("标题数不足 3")

# 8 图片
figs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
print(f"[8] 图片引用 {len(figs)}")
if len(figs) < 5:
    fails.append(f"图片数 {len(figs)}")
for f in figs:
    if not (ROOT / f).exists():
        fails.append(f"图片缺失 {f}")

# 附加
if "我" not in text:
    fails.append("缺第一人称")
if not re.search(r"[？?]", text[-400:]):
    warns.append("末尾可能缺互动钩子")

print("\n=== 结论 ===")
if warns:
    print("警告：", warns)
if fails:
    print("未通过：", fails)
    sys.exit(1)
print("全部通过")
