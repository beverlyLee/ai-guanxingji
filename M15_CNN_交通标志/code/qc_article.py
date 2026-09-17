#!/usr/bin/env python3
"""
M15 文章 QC：逐条检查用户铁律，全部基于磁盘上的真实正文与 stats.json。
用法：python code/qc_article.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = None
for p in sorted(ROOT.glob("M15_*.md")):
    ART = p
assert ART is not None, "找不到 M15 正文 md"
S = json.loads((ROOT / "data" / "stats.json").read_text(encoding="utf-8"))
text = ART.read_text(encoding="utf-8")

print("=" * 66)
print(f"M15 文章 QC 报告   正文：{ART.name}")
print("=" * 66)

# ---------- 1. 字数 ----------
body = re.sub(r"```[\s\S]*?```", "", text)
cn = len(re.findall(r"[\u4e00-\u9fff]", body))
total = len(re.sub(r"\s", "", body))
text_ns = re.sub(r"[\s,]", "", text)  # 去空白与千分位逗号，用于数字与类名比对


def has(tok):
    """正文里是否出现该标记：原样比对失败时，退到去空白/去逗号比对。"""
    if tok in text:
        return True
    return re.sub(r"[\s,]", "", tok) in text_ns


print(f"1) 字数：中文字符 {cn}，去空白总字符 {total}  -> 硬上限 8000 字")
print(f"   {'PASS' if cn <= 8000 else 'FAIL'}（中文字符数口径）")

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
# 只看真正带公式符号的段落（÷ × √ Σ x̂ σ² γ β H_out），
# 不把「卷积核」「池化」这类普通叙述词算进去，否则会大面积误报。
# 另外先剥掉「32×32」「3×3」这种尺寸写法，否则数字×数字会被当成公式。
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
R = S["runs"]
pb = S["net"]["param_breakdown"]
led = S["ledger"]
ds = S["dataset"]
scan = S["lr_scan"]

checks = [
    ("类别数", str(ds["n_class"])),
    ("训练样本数", f"{ds['n_train']}"),
    ("测试样本数", f"{ds['n_test']}"),
    ("类最多张数", f"{ds['natural_max']}"),
    ("类最少张数", f"{ds['natural_min']}"),
    ("原始训练总数", f"{ds['natural_total']}"),
    ("不平衡倍数", f"{ds['imbalance_ratio']:.0f} 倍"),
    ("输入维数", f"{led['input_dims']}"),
    ("感受野权重数", f"{led['receptive_field_dims']}"),
    ("一层卷积参数", f"{led['conv1_params']}"),
    ("等价全连接参数", f"{led['fc_equiv_params']}"),
    ("总节省倍数", f"{led['fc_equiv_ratio']:.0f}"),
    ("局部连接倍数", f"{led['local_ratio']:.2f}"),
    ("权重共享倍数", f"{led['share_ratio']:.0f}"),
    ("全网络总参", f"{pb['total']}"),
    ("卷积参数合计", f"{pb['conv']}"),
    ("BN 参数合计", f"{pb['bn']}"),
    ("全连接参数合计", f"{pb['fc']}"),
    ("卷积占比", f"{pb['conv'] / pb['total'] * 100:.1f}"),
    ("全连接占比", f"{pb['fc'] / pb['total'] * 100:.1f}"),
    ("训练轮数", str(S["net"]["epochs"])),
    ("批大小", str(S["net"]["batch"])),
    ("动量", str(S["net"]["momentum"])),
    ("扫描轮数", str(scan["epochs"])),
]
for i, r in enumerate(R):
    checks.append((f"运行{i + 1} 最终准确率", f"{r['acc_final'] * 100:.2f}"))
    checks.append((f"运行{i + 1} 学习率", f"{r['lr']:g}"))
for i, lr in enumerate(scan["lrs"]):
    checks.append((f"扫描 lr={lr:g} 带BN", f"{scan['bn'][i] * 100:.1f}"))
    checks.append((f"扫描 lr={lr:g} 无BN", f"{scan['nobn'][i] * 100:.1f}"))
tc = S["top_confusions"][0]
checks.append(("最易混张数", str(tc["count"])))
checks.append(("最易混真类", tc["true_name"]))
checks.append(("最易混预测类", tc["pred_name"]))

print("5) 正文数字 vs stats.json 一致性：")
bad = []
for label, tok in checks:
    ok = has(tok)
    if not ok:
        alt = tok.rstrip("0").rstrip(".")
        ok = bool(alt) and has(alt)
    if not ok:
        bad.append(f"{label}({tok})")
    print(f"   {'OK  ' if ok else 'MISS'} {label:22s} -> {tok}")
print(f"   {'PASS' if not bad else f'FAIL（{len(bad)} 项未命中：{bad}）'}")

# ---------- 6. 结论一致性 ----------
print("6) 关键结论核对：")
bn_scan = scan["bn"]
nobn_scan = scan["nobn"]
best_bn = max(range(len(bn_scan)), key=lambda i: bn_scan[i])
best_nobn = max(range(len(nobn_scan)), key=lambda i: nobn_scan[i])
print(f"   扫描中 BN 最优：lr={scan['lrs'][best_bn]:g} -> {bn_scan[best_bn] * 100:.1f}%")
print(f"   扫描中 无BN 最优：lr={scan['lrs'][best_nobn]:g} -> {max(nobn_scan) * 100:.1f}%")
for i, lr in enumerate(scan["lrs"]):
    mark = "BN 更高" if bn_scan[i] > nobn_scan[i] else "BN 更低"
    print(f"   lr={lr:<6g}  BN {bn_scan[i] * 100:5.1f}%  vs  无BN {nobn_scan[i] * 100:5.1f}%   {mark}")
better = sum(1 for i in range(len(scan["lrs"])) if bn_scan[i] > nobn_scan[i])
print(f"   -> {better}/{len(scan['lrs'])} 档学习率上 BN 更好  "
      f"{'PASS' if better >= len(scan['lrs']) - 1 else 'CHECK'}")
print(f"   卷积占比 {pb['conv'] / pb['total'] * 100:.1f}% vs 全连接占比 "
      f"{pb['fc'] / pb['total'] * 100:.1f}%  -> "
      f"{'PASS' if pb['conv'] < pb['fc'] else 'CHECK'}")

# 收敛速度（从 hist 现场推，正文里引用的轮数必须能对上）
print("6b) 收敛速度（全量 2x2 中的 A 与 B）：")
THR = (0.90, 0.95, 0.97, 0.98)
speed = {}
for r in R[:2]:
    accs = r["hist"]["test_acc"][1:]
    row = {}
    for th in THR:
        row[th] = next((i + 1 for i, a in enumerate(accs) if a >= th), None)
    speed[r["tag"]] = row
    print(f"   {r['tag']:16s} 首轮 {accs[0] * 100:.2f}%  末轮 {accs[-1] * 100:.2f}%  "
          + "  ".join(f">={th * 100:.0f}%:第{row[th]}轮" for th in THR))
fa = R[0]["hist"]["test_acc"]
fb = R[1]["hist"]["test_acc"]
print(f"   A 第 1 轮领先 B {abs(fa[1] - fb[1]) * 100:.1f} 个百分点"
      f"{'PASS' if has(f'{abs(fa[1] - fb[1]) * 100:.1f}') else 'CHECK'}")
print(f"   A 与 B 最终差距 {abs(fa[-1] - fb[-1]) * 100:.2f} 个百分点"
      f"{'PASS' if has(f'{abs(fa[-1] - fb[-1]) * 100:.2f}') else 'CHECK'}")
for th in THR:
    print(f"   >= {th * 100:.0f}% 所需轮数 A={speed[R[0]['tag']][th]} "
          f"B={speed[R[1]['tag']][th]}  "
          f"{'OK' if has(f'第 {speed[R[0]['tag']][th]} 轮') else 'MISS'}")

# 收敛过程中的具体数字（正文逐条引用，必须都能对上）
print("   正文引用的收敛细节：")
detail = [
    ("A 首轮准确率", f"{fa[1] * 100:.1f}"),
    ("B 首轮准确率", f"{fb[1] * 100:.1f}"),
    ("A 首轮领先", f"{abs(fa[1] - fb[1]) * 100:.1f}"),
    ("A 第 8 轮准确率", f"{fa[8] * 100:.2f}"),
    ("B 末轮准确率", f"{fb[-1] * 100:.2f}"),
    ("C 峰值准确率", f"{max(R[2]['hist']['test_acc']) * 100:.2f}"),
    ("C 峰值所在轮", str(int(max(range(len(R[2]['hist']['test_acc'])),
                                  key=lambda i: R[2]['hist']['test_acc'][i])))),
    ("D 第 11 轮准确率", f"{R[3]['hist']['test_acc'][11] * 100:.1f}"),
    ("D 末轮准确率", f"{R[3]['hist']['test_acc'][-1] * 100:.1f}"),
    ("D 第 11 轮损失", f"{R[3]['hist']['train_loss'][11]:.2f}"),
    ("D 末轮损失", f"{R[3]['hist']['train_loss'][-1]:.2f}"),
    ("BN 多出参数", f"{S['net']['bn_extra_params']}"),
    ("实验总耗时（秒）", f"{S['runtime_sec']:.0f}"),
]
miss = [f"{k}({v})" for k, v in detail if not has(v)]
for k, v in detail:
    print(f"      {'OK  ' if has(v) else 'MISS'} {k:16s} -> {v}")
print(f"   {'PASS' if not miss else f'FAIL（未命中 {miss}）'}")

# ---------- 6c. 错例分析一致性 ----------
EA_P = ROOT / "data" / "error_analysis.json"
if EA_P.exists():
    EA = json.loads(EA_P.read_text(encoding="utf-8"))
    A_ = EA["A_BN_lr0.01"]
    D_ = EA["D_noBN_lr0.05"]
    sp = EA["split"]
    print("6c) 错例分析（data/error_analysis.json）核对：")
    pairs = {f"{d['true_name']}->{d['pred_name']}": d["count"]
             for d in A_["pairs"]}
    ea_checks = [
        ("A 错例总数", str(A_["wrong_total"])),
        ("A 同族错例", str(A_["wrong_total"] - A_["cross"])),
        ("A 跨族错例", str(A_["cross"])),
        ("D 错例总数", f"{D_['wrong_total']:,}"),
        ("D 塌向限速30", f"{D_['pred_of_wrong_top'][0]['count']:,}"),
        ("D 塌向占比", f"{D_['pred_of_wrong_top'][0]['share'] * 100:.1f}"),
        ("D 跨族错例", f"{D_['cross']:,}"),
        ("D 跨族占比", f"{D_['cross'] / D_['wrong_total'] * 100:.0f}"),
        ("训练集不平衡", f"{sp['train_imbalance']:.2f}"),
        ("训练集最少每类", str(sp["train_per_class_min"])),
        ("训练集最多每类", str(sp["train_per_class_max"])),
        ("A 警告族内错", str(A_["inner"]["警告族"])),
        ("A 限速族内错", str(A_["inner"]["限速族"])),
        ("A 禁令族内错", str(A_["inner"]["禁令族"])),
        ("A 指示族内错", str(A_["inner"]["指示族"])),
    ]
    npz = np.load(ROOT / "data" / "plot_data.npz")
    cm_d = npz["cm_D_noBN_lr0.05"]
    diag_d = np.diag(cm_d)
    ea_checks += [
        ("D 判对总数", str(int(diag_d.sum()))),
        ("D 零召回类数",
         str(int(((diag_d / np.maximum(cm_d.sum(1), 1)) == 0).sum()))),
        ("D 限速30 判对数", str(int(diag_d[1]))),
    ]
    bad2 = []
    for label, tok in ea_checks:
        ok = has(tok)
        if not ok:
            alt = tok.rstrip("0").rstrip(".")
            ok = bool(alt) and has(alt)
        if not ok:
            bad2.append(f"{label}({tok})")
        print(f"   {'OK  ' if ok else 'MISS'} {label:16s} -> {tok}")
    print(f"   {'PASS' if not bad2 else f'FAIL（{len(bad2)} 项未命中：{bad2}）'}")
    wc0 = A_["wrong_classes"][0]
    print(f"   A 错例散在 {len(A_['wrong_classes'])} 个类上，"
          f"最多的一类 {wc0['name']} 错 {wc0['wrong']} 张  "
          f"{'OK' if has('限速 80') else 'MISS'}")
    for nm, cnt in list(pairs.items())[:6]:
        print(f"      配对 {nm} x{cnt}  "
              f"{'OK' if has(nm.split('->')[0][:3]) else 'MISS'}")
else:
    print("6c) 错例分析：MISS（先跑 code/analyze_errors.py）")

# ---------- 7. 结构项 ----------
print("7) 结构要求：")
print(f"   🏃 实战标记 {text.count('🏃')} 处  -> {'PASS' if text.count('🏃') >= 1 else 'FAIL'}")
n_img = len(set(re.findall(r"fig\d_\w+\.png", text)))
print(f"   引用配图 {n_img} 张  -> {'PASS' if 6 <= n_img <= 9 else 'CHECK(应 6-9)'}")
secs = re.findall(r"^## (.+)$", text, re.M)
print(f"   二级标题 {len(secs)} 个：{secs}")
print(f"   第一人称「我」 {text.count('我')} 次；"
      f"「首先」{text.count('首先')} 次、「其次」{text.count('其次')} 次")
print(f"   GitHub 链接 {text.count('github.com/beverlyLee')} 处  -> "
      f"{'PASS' if text.count('github.com/beverlyLee') >= 1 else 'FAIL'}")
print(f"   数据来源标注 {'PASS' if 'GTSRB' in text else 'CHECK'}")
print(f"   结尾互动钩子 {'PASS' if text.rstrip().endswith(('？', '?', '。')) and '评论' in text[-800:] else 'CHECK'}")

# ---------- 8. 掘金合规 ----------
print("8) 掘金合规自查：")
BAD_TITLE = ["最全", "震惊", "不看后悔", "全都在", "唯一", "史上"]
th = re.search(r"^# (.+)$", text, re.M)
title = th.group(1) if th else ""
print(f"   标题：{title}")
print(f"   绝对化词 -> {'PASS' if not any(w in title for w in BAD_TITLE) else 'FAIL'}")
print(f"   技术名词入题 {'PASS' if any(k in title for k in ['卷积', 'BatchNorm', '网络', '参数']) else 'CHECK'}")
print(f"   反问式 {'PASS' if title.rstrip().endswith(('？', '?')) else 'CHECK'}")
print(f"   二维码/加群引流 {'PASS' if '二维码' not in text and '加群' not in text else 'FAIL'}")
# 掘金第 13 章把「AI 生成直接发布」和抄袭并列，所以要有具体的第一人称痕迹：
# 失败记录 + 本机环境/耗时这类不可套模板的细节
fail_trace = [w for w in ("第一版", "踩坑", "报错", "错了", "走不通") if w in text]
env_trace = [w for w in ("numpy", "笔记本", "秒", "本机") if w in text]
print(f"   AIGC 人味改造：失败记录 {fail_trace}，本机环境/耗时 {env_trace}  -> "
      f"{'PASS' if fail_trace and env_trace else 'CHECK'}")
print("=" * 66)
