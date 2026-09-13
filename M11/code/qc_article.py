# -*- coding: utf-8 -*-
"""
M11 文章质量校验：
字数 / 第一人称 / 黑名单词 / 破折号 / 粗体 / 数学占比 / 标题/章节/配图 / 数字与 stats.json 一致。
运行：python code/qc_article.py
"""
import os
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "M11_随机森林_心脏病.md"
STATS = ROOT / "stats.json"

BLACKLIST = [
    "赋能", "洞见", "底层逻辑", "硬核", "天花板", "降维打击", "抓手",
    "闭环", "拉齐", "此外", "然而", "标志着", "至关重要", "不可或缺",
    "总而言之", "旨在",
]
# 沉淀、对齐 只在单独成词时触发
SOLO_BLACKLIST = {"沉淀", "对齐"}


def load_text():
    with open(ARTICLE, encoding="utf-8") as f:
        return f.read()


def load_stats():
    with open(STATS, encoding="utf-8") as f:
        return json.load(f)


def flatten_numbers(obj):
    """把 stats.json 里所有数字展平成字符串集合，百分比统一成小数。"""
    nums = set()
    def walk(x):
        if isinstance(x, (int, float)):
            nums.add(round(float(x), 3))
        elif isinstance(x, str):
            try:
                nums.add(round(float(x), 3))
            except ValueError:
                pass
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
    walk(obj)
    return nums


def article_numbers(text):
    """提取文章中的数字，百分比转成小数，避免把 63.2% 拆成 63 和 0.632。"""
    nums = set()
    for m in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)\s*(%)?", text):
        val = float(m.group(1))
        if m.group(2):
            val = val / 100.0
        nums.add(round(val, 3))
    return nums


def main():
    text = load_text()
    stats = load_stats()
    issues = []

    # 1. 标题
    title_match = re.search(r"^#\s+(.+)$", text, re.M)
    if not title_match:
        issues.append("未找到一级标题")
    else:
        title = title_match.group(1)
        if "随机森林" not in title:
            issues.append("标题未出现知识点关键词：随机森林")
        for w in ["最全", "震惊", "不看后悔", "绝对", "必须"]:
            if w in title:
                issues.append(f"标题含绝对化/夸张词：{w}")

    # 2. 字数
    chars = len(re.sub(r"\s", "", text))
    if chars > 8000:
        issues.append(f"字数 {chars} 超过 8000")

    # 3. 第一人称
    if text.count("我") < 1:
        issues.append("缺少第一人称")

    # 4. 黑名单词
    for w in BLACKLIST:
        if w in text:
            issues.append(f"出现黑名单词：{w}")
    for w in SOLO_BLACKLIST:
        if re.search(rf"(?<![\u4e00-\u9fa5]){w}(?![\u4e00-\u9fa5])", text):
            issues.append(f"出现黑名单词（单独使用）：{w}")

    # 5. 破折号（排除 ASCII 连字符、表格横线、代码）
    if re.search(r"[—–]|——", text):
        issues.append("存在破折号或 en/em dash")

    # 6. 粗体
    if re.search(r"\*\*.+\*\*|__.+__", text):
        issues.append("存在 markdown 粗体")

    # 7. 数学占比（希腊字母、上下标、数学符号）
    math_chars = re.findall(r"[σ²³⁰¹⁴⁵⁶⁷⁸⁹βαθπ∑∫√≈≠≤≥]|\$", text)
    math_ratio = len(math_chars) / max(chars, 1)
    if math_ratio > 0.20:
        issues.append(f"数学字符占比 {math_ratio:.2%} 超过 20%")

    # 8. 章节与配图
    headings = re.findall(r"^##\s+", text, re.M)
    figs = re.findall(r"!\[", text)
    if len(headings) < 3:
        issues.append(f"二级标题仅 {len(headings)} 个，需至少 3 个")
    if len(figs) < 5:
        issues.append(f"配图引用仅 {len(figs)} 个，需至少 5 个")
    # 配图文件是否存在
    for m in re.finditer(r"!\[.*?\]\((.+?)\)", text):
        p = ROOT / m.group(1)
        if not p.exists():
            issues.append(f"配图文件缺失：{m.group(1)}")

    # 9. 数字一致性（文章中的数字应能在 stats.json 中找到近似值）
    stats_nums = flatten_numbers(stats)
    art_nums = article_numbers(text)
    # 允许的例外：常见计数、折数、章节号、描述性区间、Python 版本号
    allowed = set(range(1, 13)) | {0.17, 0.18, 3.1}
    missing = []
    for n in art_nums:
        if n in allowed:
            continue
        # 允许 0.5% 误差
        if not any(abs(n - s) < 0.006 for s in stats_nums):
            missing.append(n)
    if missing:
        issues.append(f"文章中有 {len(missing)} 个数字在 stats.json 中找不到：{sorted(missing)[:10]}")

    # 报告
    print(f"字数：{chars}")
    print(f"二级标题：{len(headings)}")
    print(f"配图引用：{len(figs)}")
    print(f"第一人称（我）：{text.count('我')} 次")
    print(f"数学字符占比：{math_ratio:.2%}")
    if issues:
        print("\n未通过：")
        for i in issues:
            print(" -", i)
        return 1
    print("\nQC PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
