#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI观星记 CV1 文章 QC 校验脚本。
对照专栏铁律检查：中文篇幅、黑名单词(去AI味)、破折号、正文粗体、
数学占比(<=1/5)、stats.json 数字与正文一致性。
用法: python3 code/qc_article.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "CV1_OpenCV图像基本操作.md"
STATS = ROOT / "stats.json"

# AI观星记去AI味黑名单（带"深度学习/深度神经"等合法术语白名单例外）
BLACKLIST = [
    "赋能", "洞见", "底层逻辑", "硬核", "天花板", "降维打击", "抓手",
    "闭环", "沉淀", "拉齐", "此外", "然而", "标志着", "至关重要",
    "不可或缺", "总而言之", "旨在", "对齐",
]
# "深度" 单用算违规，但 "深度学习/深度神经/深度网/深度模型/深度化" 是合法术语
DEPTH_WHITELIST_NEXT = set("学神网模度化")


def count_cjk(text: str) -> int:
    return len(re.findall(r"[一-鿿]", text))


def strip_code_blocks(text: str):
    """返回 (body_without_code, code_blocks_text)。"""
    parts = re.split(r"```.*?```", text, flags=re.DOTALL)
    body = "\n".join(parts)
    code_blocks = re.findall(r"```.*?```", text, flags=re.DOTALL)
    code_text = "\n".join(code_blocks)
    return body, code_text


def check_blacklist(text: str):
    hits = []
    for w in BLACKLIST:
        for m in re.finditer(re.escape(w), text):
            hits.append((w, m.start()))
    # 单用"深度"（不在白名单术语里）
    depth_hits = []
    for m in re.finditer("深度", text):
        i = m.start()
        nxt = text[i + 2] if i + 2 < len(text) else ""
        if nxt not in DEPTH_WHITELIST_NEXT:
            depth_hits.append(i)
    return hits, depth_hits


def check_dash(text: str):
    return [m.start() for m in re.finditer("——", text)]


def check_bold(body: str):
    # 正文(去代码块)中的 markdown 粗体 **x**
    return [m.start() for m in re.finditer(r"\*\*[^*]+\*\*", body)]


# 不该出现在中文行文里的英文虚词/AI 味词（技术名词如 OpenCV/BGR 不算）
STRAY_EN = [
    "itself", "continuity", "robust", "shrink", "grow", "aggressively",
    "essentially", "basically", "actually", "however", "therefore",
    "moreover", "furthermore", "simply", "obviously",
]


def check_stray_english(body: str):
    hits = []
    for w in STRAY_EN:
        for m in re.finditer(rf"\b{w}\b", body, flags=re.IGNORECASE):
            hits.append((w, m.start()))
    return hits


def estimate_math_ratio(text: str):
    """估算公式"墨水"占比：只统计公式区里的数学符号与公式内中文，
    不把公式里的英文字母/数字/标点算作数学，更贴近"数学密度"的真实含义。"""
    MATH_SYMBOLS = set("Σ∂σ√≈²π×·−αβγμωλθΔ∇≤≥∫∑∏⊥∈∀∃^⁺⁻")
    blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    inlines = re.findall(r"`([^`]+)`", text)
    formula_regions = []
    for b in blocks:
        if re.search(r"[Σ∂σ√≈²αβγμωλθΔ∇]|exp|sqrt", b):
            formula_regions.append(b)
    for s in inlines:
        if re.search(r"[=∂σ√≈²Σ]|exp|sqrt", s):
            formula_regions.append(s)
    math_chars = 0
    for r in formula_regions:
        for ch in r:
            if ("一" <= ch <= "鿿") or ch in MATH_SYMBOLS:
                math_chars += 1
    total_cjk = count_cjk(text)
    ratio = (math_chars / total_cjk) if total_cjk else 0
    return math_chars, total_cjk, ratio


def check_stats_consistency(article_text: str, stats: dict):
    missing = []
    # 标量数字
    scalar_keys = [
        "page_mean_std", "threshold_global_val", "threshold_otsu_val",
        "coins_fg_ratio_global", "coins_fg_ratio_otsu", "page_otsu_val",
        "denoise_noise_std_raw", "denoise_std_mean", "denoise_std_gauss",
        "denoise_std_median", "denoise_clean_page_std",
        "morph_components_raw", "morph_components_open", "morph_components_close",
        "sobel_edge_pixel_ratio", "sobel_mag_max", "laplacian_max",
    ]
    for k in scalar_keys:
        v = stats.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            # 数组型（如 page_mean_std=[171.54, 56.81]）逐个元素检查
            for elem in v:
                se = str(elem)
                cands = {se}
                if isinstance(elem, float) and elem == int(elem):
                    cands.add(str(int(elem)))
                if not any(c in article_text for c in cands):
                    missing.append((f"{k}[]", elem))
            continue
        sval = str(v)
        # 浮点去掉多余.0
        candidates = {sval}
        if isinstance(v, float):
            candidates.add(str(int(v)) if v == int(v) else sval)
        if not any(c in article_text for c in candidates):
            missing.append((k, v))
    # shapes：检查 "H×W" 与 "H, W" 两种写法
    shapes = stats.get("shapes", {})
    shape_reprs = {
        "page": "191×384", "camera": "512×512", "coins": "303×384",
        "astronaut": "512×512×3",
    }
    for key, want in shape_reprs.items():
        if want not in article_text:
            missing.append((f"shapes.{key}", want))
    # 百分比原值
    pct_map = {
        "coins_fg_ratio_global": "0.2962",
        "coins_fg_ratio_otsu": "0.3878",
        "sobel_edge_pixel_ratio": "0.0837",
    }
    for k, raw in pct_map.items():
        if raw not in article_text:
            missing.append((f"{k}(原值)", raw))
    return missing


def main():
    text = ARTICLE.read_text(encoding="utf-8")
    body, code_text = strip_code_blocks(text)
    stats = json.loads(STATS.read_text(encoding="utf-8"))

    total_cjk = count_cjk(text)
    body_cjk = count_cjk(body)
    blacklist_hits, depth_hits = check_blacklist(text)
    dash_hits = check_dash(text)
    bold_hits = check_bold(body)
    stray_hits = check_stray_english(body)
    math_chars, mc_total, math_ratio = estimate_math_ratio(text)
    missing_stats = check_stats_consistency(text, stats)

    report = []
    report.append("===== CV1 文章 QC 报告 =====")
    report.append(f"1. 中文篇幅: 全文中文字={total_cjk}  去代码块后中文={body_cjk}  (目标≈10000)")
    report.append(f"2. 黑名单词: {len(blacklist_hits)} 处  -> {[w for w,_ in blacklist_hits]}")
    report.append(f"   单用'深度'(非深度学习/神经): {len(depth_hits)} 处")
    report.append(f"3. 破折号 —— : {len(dash_hits)} 处")
    report.append(f"4. 正文粗体 ** : {len(bold_hits)} 处")
    report.append(f"5. 数学占比: 公式字符≈{math_chars} / 中文{body_cjk} = {math_ratio:.1%}  (上限 20%)")
    report.append(f"6. stats.json 数字一致性: 缺失 {len(missing_stats)} 项 -> {missing_stats}")
    report.append(f"7. 中文行文夹带英文虚词: {len(stray_hits)} 处 -> {[w for w,_ in stray_hits]}")
    report.append("==============================")

    # 判定
    fail = []
    if total_cjk < 8000:
        fail.append("中文字数不足 8000")
    if blacklist_hits or depth_hits:
        fail.append("存在黑名单词")
    if dash_hits:
        fail.append("存在破折号")
    if bold_hits:
        fail.append("存在正文粗体")
    if math_ratio > 0.20:
        fail.append("数学占比超 1/5")
    if missing_stats:
        fail.append("stats 数字未全部出现在正文")
    if stray_hits:
        fail.append("中文行文夹带英文虚词")

    if fail:
        report.append(f"[FAIL] 未通过: {fail}")
    else:
        report.append("[PASS] 全部通过")
    print("\n".join(report))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
