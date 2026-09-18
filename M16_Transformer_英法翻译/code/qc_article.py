#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_article.py — M16 文章 QC 八项校验
1) 字数  2) 黑名单词  3) 破折号  4) 粗体  5) 数学占比≤20%
6) 标题数  7) 图片数(要求 8 张)  8) 数字一致性(对照 stats_train.json / stats_data.json)
根目录 = 本文件上级的上级
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

ARTICLE = ROOT / "M16_Transformer_英法翻译.md"

# 系列黑名单（AI 味词簇）
BANNED = ["赋能", "深度", "洞见", "底层逻辑", "硬核", "天花板", "降维打击", "抓手",
          "闭环", "沉淀", "拉齐", "对齐", "此外", "然而", "标志着", "至关重要",
          "不可或缺", "总而言之", "旨在"]

# 数学符号集合（仅认这些，不认尺寸/字号写法）
MATH_SYMBOLS = set("×÷√ΣΠ∫≈≠≤≥±→∈∉⊂⊃∀∃∇∂∞∑∏·^²³⁴₁₂₃₄"
                   "αβγδθλμπσφψωΑΒΓΔΘΛΞΠΣΦΨΩ")

def load_text():
    return ARTICLE.read_text(encoding="utf-8")


def main():
    text = load_text()
    total = len(text)
    fails, warns, passes = [], [], []

    # 1) 字数
    if total > 10500:
        fails.append(f"字数 {total} 超过约 10000 上限（M16 放宽到≈10000）")
    else:
        passes.append(f"字数 {total}（目标≈10000，上限 10500）")

    # 2) 黑名单词
    for w in BANNED:
        if w in text:
            fails.append(f"黑名单词出现：{w}")
    if not any(w in text for w in BANNED):
        passes.append("黑名单词 0 处")

    # 3) 破折号
    if "——" in text or "—" in text or "–" in text:
        fails.append("出现破折号/连字符长横（—— 或 —），需删除")
    else:
        passes.append("无破折号")

    # 4) 粗体
    if re.search(r"\*\*[^*]+\*\*", text):
        fails.append("出现 Markdown 粗体 **...**，需删除")
    else:
        passes.append("无粗体标记")

    # 5) 数学占比
    math_chars = 0
    # 行内/块级公式
    for m in re.findall(r"\$+([^$]+?)\$+", text, flags=re.S):
        math_chars += len(m)
    # 公式外的数学符号
    for ch in text:
        if ch in MATH_SYMBOLS:
            math_chars += 1
    ratio = math_chars / total if total else 0
    if ratio > 0.20:
        fails.append(f"数学字符占比 {ratio:.1%} 超过 20%")
    else:
        passes.append(f"数学占比 {ratio:.1%}（≤20%）")

    # 6) 标题数
    heads = re.findall(r"^#{1,6}\s+.+$", text, flags=re.M)
    if len(heads) < 6:
        warns.append(f"标题数 {len(heads)} 偏少（建议≥6，便于结构清晰）")
    else:
        passes.append(f"标题数 {len(heads)}")

    # 7) 图片数
    figs = sorted(p.name for p in FIG.glob("fig_*.png")) if FIG.exists() else []
    if len(figs) < 8:
        warns.append(f"配图 {len(figs)} 张（要求 8 张），缺 {8 - len(figs)} 张")
    else:
        passes.append(f"配图 {len(figs)} 张（≥8）")

    # 8) 数字一致性
    # 载入两份 stats
    consistency = []
    try:
        st = json.loads((DATA / "stats_train.json").read_text(encoding="utf-8"))
        sd = json.loads((DATA / "stats_data.json").read_text(encoding="utf-8"))
        hp = st["hyperparams"]
        pl = st["param_ledger"]
        tr = st["training"]
        # 超参对照
        checks = [
            ("d_model", hp["d_model"]), ("n_head", hp["n_head"]),
            ("n_layer", hp["n_layer"]), ("d_ff", hp["d_ff"]),
            ("epochs", hp["epochs"]), ("train_size", hp["train_size"]),
            ("test_size", hp["test_size"]),
            ("src_vocab", hp["src_vocab"]), ("tgt_vocab", hp["tgt_vocab"]),
            ("total_params", pl["total"]), ("device", hp["device"]),
            ("final_test_ppl", tr["final_test_ppl"]),
            ("test_bleu", tr["test_bleu_1gram"]),
        ]
        for name, val in checks:
            # 在文章里查找这个值（数字形式）
            if name in ("device",):
                continue
            if name == "total_params":
                # 允许 8.79M / 8790000 / 8,787,776 等多种写法
                candidates = [f"{val:,}", f"{val/1e6:.2f}M", f"{val/1e6:.2f} M",
                              str(val), f"{val/1e6:.1f}M"]
                found = any(c in text for c in candidates)
            else:
                candidates = [str(val), f"{val:,}"]
                found = any(c in text for c in candidates)
            if found:
                consistency.append(f"  ✓ {name}={val}")
            else:
                consistency.append(f"  ✗ {name}={val} 未在文中出现（可能需补/核对）")
        # 数据侧：raw/used（stats_data.json 实际嵌套在 dataset 下）
        ds = sd.get("dataset", sd)
        for name, val in [("pairs_raw", ds.get("pairs_raw")),
                          ("pairs_used", ds.get("pairs_used"))]:
            if val and (str(val) in text or f"{val:,}" in text):
                consistency.append(f"  ✓ {name}={val}")
            else:
                consistency.append(f"  ~ {name}={val}（可选，未强制）")
    except FileNotFoundError as e:
        warns.append(f"数字一致性未校验：缺少 stats 文件 {e}")

    # ---- 汇总 ----
    print("=" * 60)
    print(f"M16 文章 QC 报告（字数 {total}）")
    print("=" * 60)
    for p in passes:
        print("PASS |", p)
    for w in warns:
        print("WARN |", w)
    for f in fails:
        print("FAIL |", f)
    print("---- 数字一致性 ----")
    for c in consistency:
        print(c)
    print("=" * 60)
    if fails:
        print(f"结论：{len(fails)} 项 FAIL，需修改后重跑")
    else:
        print(f"结论：无 FAIL（WARN {len(warns)} 项），可进入推仓")


if __name__ == "__main__":
    main()
