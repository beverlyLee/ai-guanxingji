# -*- coding: utf-8 -*-
"""CV5 文章 QC：字数、黑名单、破折号、粗体、数学占比、stats 一致性、图片数。"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "正文.md"
STATS = ROOT / "stats.json"

BLACKLIST = [
    "赋能", "洞见", "底层逻辑", "硬核", "天花板", "降维打击",
    "抓手", "闭环", "沉淀", "拉齐", "对齐", "此外", "然而",
    "标志着", "至关重要", "不可或缺", "总而言之", "旨在",
]

PICTURES = [
    "fig1_macro.png",
    "fig2_lenet.png",
    "fig3_vgg_stack.png",
    "fig4_inception.png",
    "fig5_resnet.png",
    "fig6_rf.png",
]


def count_chinese(text: str) -> int:
    """统计中文字符数量（含中文标点）。"""
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def extract_math(text: str) -> str:
    """提取 $...$ 和 $$...$$ 之间的内容。"""
    # 先处理 $$...$$
    out = ""
    for block in re.findall(r"\$\$(.*?)\$\$", text, flags=re.S):
        out += block
    # 再处理 $...$，避免和 $$ 重叠
    text2 = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    for block in re.findall(r"\$(.*?)\$", text2, flags=re.S):
        out += block
    return out


def main():
    if not ARTICLE.exists():
        print(f"FAIL: 正文不存在 {ARTICLE}", file=sys.stderr)
        sys.exit(1)
    text = ARTICLE.read_text(encoding="utf-8")

    stats = json.loads(STATS.read_text(encoding="utf-8"))

    fails = []

    # 1. 字数
    cn = count_chinese(text)
    print(f"中文字符数: {cn}")
    if not (8500 <= cn <= 12500):
        fails.append(f"字数 {cn} 不在 8500-12500 区间")

    # 2. 黑名单
    hits = [w for w in BLACKLIST if w in text]
    if hits:
        fails.append(f"黑名单词: {hits}")
    print(f"黑名单词命中: {len(hits)}")

    # 3. 破折号
    em_dash = text.count("——")
    print(f"破折号(——)数量: {em_dash}")
    if em_dash:
        fails.append(f"破折号 {em_dash} 处未删除")

    # 4. 粗体
    bold = re.findall(r"\*\*[^*]+\*\*", text)
    print(f"粗体(**)数量: {len(bold)}")
    if bold:
        fails.append(f"粗体 {len(bold)} 处未删除: {bold[:3]}")

    # 5. 数学占比
    math_text = extract_math(text)
    math_chars = len(re.sub(r"\s", "", math_text))
    total_chars = count_chinese(text)
    ratio = math_chars / total_chars if total_chars else 0
    print(f"数学字符数(不含空格): {math_chars}; 占比: {ratio*100:.2f}%")
    if ratio > 0.20:
        fails.append(f"数学占比 {ratio*100:.2f}% > 20%")

    # 6. 图片引用
    missing_figs = [f for f in PICTURES if f not in text]
    print(f"图片引用缺失: {missing_figs if missing_figs else '无'}")
    if missing_figs:
        fails.append(f"缺少图片引用: {missing_figs}")

    # 7. 关键数字一致性（从 stats.json 提取，按正文展示字符串匹配）
    expected = {
        "6.2 万": "0.0617M" in json.dumps(stats),
        "61706": str(stats["architectures"]["LeNet-5"]["params"]) == "61706",
        "6097 万": str(stats["architectures"]["AlexNet"]["params"]) == "60965224",
        "138,357,544": str(stats["architectures"]["VGG16"]["params"]) == "138357544",
        "1.38 亿": str(stats["architectures"]["VGG16"]["params_M"]) == "138.3575",
        "699.8 万": str(stats["architectures"]["GoogLeNet(Inception-v1)"]["params"]) == "6998216",
        "25,557,032": str(stats["architectures"]["ResNet-50"]["params"]) == "25557032",
        "62.9%": stats["architectures"]["AlexNet"]["imagenet_top1"] == 0.629,
        "71.5%": stats["architectures"]["VGG16"]["imagenet_top1"] == 0.715,
        "69.3%": stats["architectures"]["GoogLeNet(Inception-v1)"]["imagenet_top1"] == 0.693,
        "75.3%": stats["architectures"]["ResNet-50"]["imagenet_top1"] == 0.753,
        "7.24%": stats["degradation"]["plain_20_layer_train_err"] == 0.0724,
        "9.97%": stats["degradation"]["plain_56_layer_train_err"] == 0.0997,
        "6.41%": stats["degradation"]["resnet_56_layer_train_err"] == 0.0641,
        "212": stats["receptive_field"]["VGG16_rf_at_stage_pools"]["final"] == 212,
        "427": stats["receptive_field"]["ResNet-50_final_rf"] == 427,
    }

    # 这里我们反向检查：正文是否包含 stats 里的关键数字
    key_numbers = []
    for arch, data in stats["architectures"].items():
        # 只检查 stats.json 里存的原始字符串（不额外做舍入，避免假失败）
        key_numbers.append(str(data["params_M"]))
        if data["imagenet_top1"] is not None:
            key_numbers.append(f"{data['imagenet_top1']*100:.1f}")
    key_numbers.append(f"{stats['degradation']['plain_20_layer_train_err']*100:.2f}")
    key_numbers.append(f"{stats['degradation']['plain_56_layer_train_err']*100:.2f}")
    key_numbers.append(f"{stats['degradation']['resnet_20_layer_train_err']*100:.2f}")
    key_numbers.append(f"{stats['degradation']['resnet_56_layer_train_err']*100:.2f}")

    missing_in_text = [n for n in key_numbers if n not in text]
    if missing_in_text:
        fails.append(f"stats 关键数字未在正文出现: {missing_in_text}")
    print(f"stats 关键数字正文缺失: {missing_in_text if missing_in_text else '无'}")

    # 8. 代码块存在
    if "python experiment.py" not in text:
        fails.append("缺少复现代码块引用")
    print(f"复现代码块引用: {'有' if 'python experiment.py' in text else '无'}")

    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)

    print("\n全部 QC 通过")


if __name__ == "__main__":
    main()
