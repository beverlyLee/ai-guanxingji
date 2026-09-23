#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV4 文章 QC：逐条检查用户铁律，全部基于磁盘上的真实正文与 stats.json。
用法：python code/qc_article.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = None
for p in sorted(ROOT.glob("*.md")):
    ART = p
assert ART is not None, "找不到 CV4 正文 md"
S = json.loads((ROOT / "stats.json").read_text(encoding="utf-8"))
text = ART.read_text(encoding="utf-8")

print("=" * 68)
print(f"CV4 文章 QC 报告   正文：{ART.name}")
print("=" * 68)

# ---------- 1. 字数 ----------
body = re.sub(r"```[\s\S]*?```", "", text)
cn = len(re.findall(r"[\u4e00-\u9fff]", body))
total = len(re.sub(r"\s", "", body))
text_ns = re.sub(r"[\s,]", "", text)


def has(tok):
    if tok in text:
        return True
    return re.sub(r"[\s,]", "", tok) in text_ns


print(f"1) 字数：中文字符 {cn}，去空白总字符 {total}  -> 目标约 10000 字")
print(f"   {'PASS' if 8500 <= cn <= 12500 else 'CHECK'}（目标区间 8500-12500）")

# ---------- 2. 黑名单词 ----------
BLACK = ["赋能", "洞见", "底层逻辑", "硬核", "天花板", "降维打击", "抓手",
         "闭环", "沉淀", "拉齐", "对齐", "此外", "然而", "标志着", "至关重要",
         "不可或缺", "总而言之", "旨在", "综上所述", "值得注意的是",
         "一方面", "无疑是", "显而易见"]
hit = {w: text.count(w) for w in BLACK if w in text}
print(f"2) 黑名单词：{hit if hit else '无'}  -> {'PASS' if not hit else 'FAIL'}")
shen_all = text.count("深度")
shen_ok = text.count("深度学习")
print(f"   「深度」共 {shen_all} 次，其中「深度学习」{shen_ok} 次，"
      f"其余 {shen_all - shen_ok} 次  -> "
      f"{'PASS' if shen_all - shen_ok == 0 else 'FAIL'}")

# ---------- 3. 破折号 / 粗体 ----------
prose = "\n".join(l for l in text.split("\n") if not l.strip().startswith("|"))
dash = len(re.findall(r"——", prose))
bold = len(re.findall(r"\*\*", prose)) // 2
print(f"3) 正文破折号 {dash} 处，粗体标记 {bold} 处  -> "
      f"{'PASS' if dash == 0 and bold == 0 else 'FAIL'}")

# ---------- 4. 数学占比 ----------
FORMULA_TOKENS = ["÷", "×", "√", "Σ", "x̂", "σ²", "γ", "β", "H_out", "·"]
SIZE_RE = re.compile(r"(?<![\d.])\d+\s*×\s*\d+(?![\d.])")
paras = [p for p in prose.split("\n") if p.strip() and not p.strip().startswith("#")]
math_paras = [p for p in paras
              if any(t in SIZE_RE.sub("", p) for t in FORMULA_TOKENS)]
math_cn = sum(len(re.findall(r"[\u4e00-\u9fff]", p)) for p in math_paras)
ratio = math_cn / max(cn, 1)
print(f"4) 带公式符号的段落 {len(math_paras)}/{len(paras)}，占中文 {math_cn} 字"
      f"（{ratio * 100:.1f}%）  -> 目标 ≤ 20%  {'PASS' if ratio <= 0.20 else 'FAIL'}")
if ratio > 0.20:
    for p in math_paras:
        print(f"      · {p[:60]}")

# ---------- 5. 数字一致性 ----------
mn = S["mnist"]
cc = S["credit_card"]
pk = S["parking"]
pn = S["panorama"]
wd = cc["wrong_digits"][0]

checks = [
    ("训练样本数", f"{mn['n_train']}"),
    ("测试样本数", f"{mn['n_test']}"),
    ("MLP 准确率", f"{mn['mlp_test_acc'] * 100:.2f}"),
    ("最少类张数(5)", f"{mn['class_counts']['5']}"),
    ("最多类张数(1)", f"{mn['class_counts']['1']}"),
    ("训练子集", "20000"),
    ("训练轮数", "20"),
    ("输入维数", "784"),
    ("卡面真值位数", f"{cc['n_digits_truth']}"),
    ("卡面检测位数", f"{cc['n_digits_detected']}"),
    ("识别正确位数", f"{cc['n_recognized_correct']}"),
    ("卡面准确率", f"{cc['card_accuracy'] * 100:.2f}"),
    ("Otsu 阈值", f"{cc['otsu_threshold']}"),
    ("错位真值", f"{wd['gt']}"),
    ("错位预测", f"{wd['pred']}"),
    ("框宽高比", f"{cc['box_ar_min']:.1f}"),
    ("车位帧数", f"{pk['n_frames']}"),
    ("每帧车位数", f"{pk['bays_per_frame']}"),
    ("真占用总数", f"{pk['n_gt_total']}"),
    ("判对总数", f"{pk['tp']}"),
    ("误报数", f"{pk['fp']}"),
    ("漏报数", f"{pk['fn']}"),
    ("精确率", f"{pk['precision']:.4f}"),
    ("召回率", f"{pk['recall']:.4f}"),
    ("F1", f"{pk['f1']:.4f}"),
    ("车位框宽", f"{pk['bay_box_w']}"),
    ("车位框高", f"{pk['bay_box_h']}"),
    ("占用块宽", f"{pk['occ_block_w']}"),
    ("占用块高", f"{pk['occ_block_h']}"),
    ("占用块面积", f"{pk['occ_block_area']}"),
    ("面积阈值", f"{pk['area_thresh']}"),
    ("关键点图1", f"{pn['n_keypoints_img1']}"),
    ("关键点图2", f"{pn['n_keypoints_img2']}"),
    ("比值检验对数", f"{pn['n_matches_ratio_test']}"),
    ("RANSAC 内点", f"{pn['n_ransac_inliers']}"),
    ("内点率", f"{pn['inlier_ratio']:.4f}"),
    ("平均重投影误差", f"{pn['mean_reproj_error_px']:.3f}"),
    ("与真值差距", f"{pn['recovered_vs_gt_homography_err_px']:.2f}"),
    ("RANSAC 阈值", f"{pn['ransac_threshold_px']:.0f}"),
    ("旋转角", f"{pn['rotation_deg']:.0f}"),
    ("缩放", f"{pn['scale']:.2f}"),
    ("平移 x", f"{pn['translation_px'][0]}"),
    ("平移 y", f"{pn['translation_px'][1]}"),
    ("比值阈值", f"{pn['ratio_test_thresh']:.1f}"),
    ("描述子维数", f"{pn['sift_descriptor_dim']}"),
    ("单应自由度", f"{pn['homography_dof']}"),
]
print("5) 正文数字 vs stats.json 一致性：")
bad = []
for label, tok in checks:
    ok = has(tok)
    if not ok:
        alt = tok.rstrip("0").rstrip(".")
        ok = bool(alt) and has(alt)
    if not ok:
        bad.append(f"{label}({tok})")
    print(f"   {'OK  ' if ok else 'MISS'} {label:16s} -> {tok}")
print(f"   {'PASS' if not bad else f'FAIL（{len(bad)} 项未命中：{bad}）'}")

# ---------- 6. 派生数字一致性 ----------
print("6) 派生数字核对（正文里算出来的数，必须能被 stats 复现）：")
derived = [
    ("车位样本总数", str(pk["bays_per_frame"] * pk["n_frames"]),
     f"{pk['bays_per_frame'] * pk['n_frames']}"),
    ("占用占比", f"{pk['n_gt_total'] / (pk['bays_per_frame'] * pk['n_frames']) * 100:.1f}",
     f"{pk['n_gt_total'] / (pk['bays_per_frame'] * pk['n_frames']) * 100:.1f}"),
    ("面积阈值占理论面积", f"{pk['area_thresh'] / pk['occ_block_area'] * 100:.1f}",
     f"{pk['area_thresh'] / pk['occ_block_area'] * 100:.1f}"),
    ("16 位期望错数", f"{cc['n_digits_truth'] * (1 - mn['mlp_test_acc']):.2f}",
     f"{cc['n_digits_truth'] * (1 - mn['mlp_test_acc']):.2f}"),
    ("召回率算式", f"{pk['tp']} / {pk['n_gt_total']}", f"{pk['recall']:.4f}"),
    ("内点率算式", f"{pn['n_ransac_inliers']} 对", f"{pn['inlier_ratio']:.4f}"),
]
for label, tok, _v in derived:
    ok = has(tok)
    if not ok:
        alt = tok.rstrip("0").rstrip(".")
        ok = bool(alt) and has(alt)
    print(f"   {'OK  ' if ok else 'MISS'} {label:18s} -> {tok}")

# RANSAC 迭代次数表逐项核对
print("6b) RANSAC 迭代次数表（N = log(1-p)/log(1-w^4)，p=0.99）：")
miss_tbl = []
for w, n in pn["ransac_iters_table"]:
    ws = f"{w:.4f}".rstrip("0").rstrip(".") if w < 1 else f"{w:.4f}"
    ws = f"{w:g}"
    tok = f"{n} 轮"
    okn = has(tok)
    okw = has(ws)
    if not (okn and okw):
        miss_tbl.append(f"w={ws}->{tok}")
    print(f"   {'OK  ' if (okn and okw) else 'MISS'} 内点率 {ws:8s} -> {tok:8s}")
lo = pn["ransac_iters_table"][0][1]
hi = pn["ransac_iters_table"][-1][1]
ratio_x = lo / hi
print(f"   最低档 {lo} 轮 vs 最高档 {hi} 轮，倍数 {ratio_x:.1f}  "
      f"{'OK' if has('190') or has(f'{ratio_x:.0f}') else 'MISS'}")
print(f"   {'PASS' if not miss_tbl else f'FAIL（{miss_tbl}）'}")

# 关键结论：precision=1 且 fp=0，recall<1 且 fn>0，且 p 高于 r
print("6c) 关键结论核对：")
print(f"   精确率 {pk['precision']} 且误报 {pk['fp']} -> "
      f"{'PASS' if pk['precision'] == 1.0 and pk['fp'] == 0 else 'CHECK'}")
print(f"   召回率 {pk['recall']} < 1 且漏报 {pk['fn']} > 0 -> "
      f"{'PASS' if pk['recall'] < 1 and pk['fn'] > 0 else 'CHECK'}")
print(f"   误差两项不同（自洽 {pn['mean_reproj_error_px']} vs 正确 "
      f"{pn['recovered_vs_gt_homography_err_px']}）-> "
      f"{'PASS' if pn['mean_reproj_error_px'] != pn['recovered_vs_gt_homography_err_px'] else 'CHECK'}")
print(f"   内点率 0.9559 是上界已声明 -> "
      f"{'PASS' if '上界' in text else 'FAIL'}")

# ---------- 7. 结构项 ----------
print("7) 结构要求：")
print(f"   🏃 实战标记 {text.count('🏃')} 处  -> {'PASS' if text.count('🏃') >= 1 else 'FAIL'}")
imgs = sorted(set(re.findall(r"fig\d\w*_\w+\.png", text)))
print(f"   引用配图 {len(imgs)} 张：{imgs}  -> "
      f"{'PASS' if 6 <= len(imgs) <= 9 else 'CHECK(应 6-9)'}")
missing = [p.name for p in (ROOT / "figures").glob("*.png")
           if p.name not in imgs and p.name != "fig5b_pr.png"] + \
          [n for n in imgs if not (ROOT / "figures" / n).exists()]
print(f"   图文件与引用一致性 -> {'PASS' if not missing else f'FAIL（{missing}）'}")
secs = re.findall(r"^## (.+)$", text, re.M)
print(f"   二级标题 {len(secs)} 个：")
for s in secs:
    print(f"      · {s}")
print(f"   第一人称「我」 {text.count('我')} 次；"
      f"「首先」{text.count('首先')} 次、「其次」{text.count('其次')} 次")
print(f"   GitHub 链接 {text.count('github.com/beverlyLee')} 处  -> "
      f"{'PASS' if text.count('github.com/beverlyLee') >= 1 else 'FAIL'}")
print(f"   数据来源标注 MNIST {'OK' if 'MNIST' in text else 'MISS'} / "
      f"building.jpg {'OK' if 'building.jpg' in text else 'MISS'} / "
      f"PKLot {'OK' if 'PKLot' in text else 'MISS'}")
# 三个干货点
gd = re.search(r"## 三个值得记住的点([\s\S]*?)\n##", text)
n_gd = len(re.findall(r"^\d+\. ", gd.group(1), re.M)) if gd else 0
print(f"   干货点数量 {n_gd}  -> {'PASS' if n_gd == 3 else 'FAIL'}")
tail = text.rstrip()[-900:]
print(f"   结尾互动钩子 -> "
      f"{'PASS' if ('评论' in tail and tail.rstrip().endswith(('？', '?', '。'))) else 'CHECK'}")

# ---------- 8. 掘金合规 ----------
print("8) 掘金合规自查：")
BAD_TITLE = ["最全", "震惊", "不看后悔", "全都在", "唯一", "史上"]
th = re.search(r"^# (.+)$", text, re.M)
title = th.group(1) if th else ""
print(f"   标题：{title}")
print(f"   绝对化词 -> {'PASS' if not any(w in title for w in BAD_TITLE) else 'FAIL'}")
print(f"   技术名词入题 -> "
      f"{'PASS' if any(k in title for k in ['OpenCV', '图像处理', 'Python']) else 'FAIL'}")
print(f"   疑问句收尾 -> {'PASS' if title.rstrip().endswith(('？', '?')) else 'CHECK'}")
print(f"   字数 >= 250 -> {'PASS' if cn >= 250 else 'FAIL'}")
print(f"   二维码/加群引流 -> "
      f"{'PASS' if '二维码' not in text and '加群' not in text else 'FAIL'}")
print(f"   不画中国地图/不涉省际对比 -> "
      f"{'PASS' if not any(w in text for w in ['中国地图', '省份排名', '各省']) else 'CHECK'}")
fail_trace = [w for w in ("第一版", "踩坑", "报错", "错了", "走不通", "后怕") if w in text]
env_trace = [w for w in ("numpy", "笔记本", "秒", "OpenCV 5.0") if w in text]
print(f"   AIGC 人味改造：失败记录 {fail_trace}，本机环境/耗时 {env_trace}  -> "
      f"{'PASS' if fail_trace and env_trace else 'CHECK'}")
print("=" * 68)
