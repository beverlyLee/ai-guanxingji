"""M14 文章 QC：逐条检查用户铁律，基于磁盘上的真实正文与 stats.json。"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "M14_CNN_人脸_卷积与BN_2026-09-16.md"
S = json.loads((ROOT / "data" / "stats.json").read_text(encoding="utf-8"))
text = ART.read_text(encoding="utf-8")

print("=" * 62)
print("M14 文章 QC 报告")
print("=" * 62)

# ---------- 1. 字数 ----------
body = re.sub(r"```[\s\S]*?```", "", text)
body = re.sub(r"图片占位.*", "", body)
cn = len(re.findall(r"[\u4e00-\u9fff]", body))
total = len(re.sub(r"\s", "", body))
print(f"1) 字数：中文字符 {cn}，去空白总字符 {total}  -> 硬上限 10000（用户要求精简）")
print(f"   {'PASS' if 5000 <= total <= 10000 else 'CHECK（目标区间 5000-10000）'}")

# ---------- 2. 黑名单词 ----------
BLACK = ["赋能", "洞见", "底层逻辑", "硬核", "天花板", "降维打击", "抓手",
         "闭环", "沉淀", "拉齐", "对齐", "此外", "然而", "标志着", "至关重要",
         "不可或缺", "总而言之", "旨在"]
hit = {w: text.count(w) for w in BLACK if w in text}
# 深度 单独处理：只允许出现在"深度学习"这个专有名词里
shen_ok = text.count("深度学习")
shen_all = text.count("深度")
shen_bad = shen_all - shen_ok
print(f"2) 黑名单词：{hit if hit else '无'}  -> {'PASS' if not hit else 'FAIL'}")
print(f"   「深度」出现 {shen_all} 次，其中「深度学习」{shen_ok} 次，"
      f"其余 {shen_bad} 次  -> {'PASS' if shen_bad == 0 else 'FAIL'}")

# ---------- 3. 破折号 / 粗体 ----------
prose = "\n".join(l for l in text.split("\n") if not l.strip().startswith("|"))
dash = len(re.findall(r"——", prose))
bold = len(re.findall(r"\*\*", prose))
print(f"3) 正文破折号 {dash} 处，粗体标记 {bold // 2} 处  -> {'PASS' if dash == 0 and bold == 0 else 'FAIL'}")

# ---------- 4. 数学占比 ----------
MATH_TERMS = ["y[i, j]", "ΣΣ", "H_out", "x̂ =", "y = γ", "dx =", "im2col",
              "感受野", "权值共享", "卷积核", "方差", "均值", "偏置"]
paras = [p for p in prose.split("\n") if p.strip() and not p.strip().startswith("#")]
math_paras = [p for p in paras if sum(t in p for t in MATH_TERMS) >= 2]
math_cn = sum(len(re.findall(r"[\u4e00-\u9fff]", p)) for p in math_paras)
ratio = math_cn / max(cn, 1)
print(f"4) 讲公式/机制的段落 {len(math_paras)}/{len(paras)}，占中文字数 {math_cn}"
      f"（{ratio * 100:.1f}%）  -> 目标 ≤ 20%  {'PASS' if ratio <= 0.20 else 'FAIL'}")

# ---------- 5. 数字一致性：正文 vs stats.json ----------
checks = [
    ("训练样本 300", "300", S["n_train"]),
    ("测试样本 100", "100", S["n_test"]),
    ("卷积参数量 88", "88", S["conv1_params"]),
    ("等价全连接 32776", "32776", S["equiv_fc1_params"]),
    ("减少倍数 409.7", "409.7", S["conv_param_reduction_x"]),
    ("全网络总参 9442", "9442", S["total_params"]),
    ("带 BN 准确率 0.95", "0.95", S["acc_with_bn"]),
    ("不带 BN 准确率 0.65", "0.65", S["acc_without_bn"]),
    ("不带 BN 大 lr 0.78", "0.78", S["acc_without_bn_biglr"]),
    ("带 BN 终 loss 0.0033", "0.0033", S["loss_bn"][-1]),
    ("不带 BN 终 loss 0.3466", "0.3466", S["loss_without_bn"][-1]),
    ("带 BN 时 148.7s", "148.7", S["time_with_bn_s"]),
    ("不带 BN 时 151.1s", "151.1", S["time_without_bn_s"]),
]
print("5) 正文数字 vs stats.json 一致性：")
bad = 0
for label, _, val in checks:
    tok = str(val)
    ok = tok in text or tok.rstrip("0") in text
    if not ok and isinstance(val, float):
        ok = f"{val:.1f}" in text or f"{val:.0f}" in text
    if not ok:
        bad += 1
    print(f"   {'OK  ' if ok else 'MISS'} {label:28s} -> {val}")
print(f"   {'PASS' if bad == 0 else f'FAIL（{bad} 项未在正文命中）'}")

# ---------- 6. 混淆矩阵一致性 ----------
cm = S["cm_with_bn"]
print(f"6) 混淆矩阵 stats = {cm}  -> 正文写 48/2/3/47")
tok = ["48", "2", "3", "47"]
print(f"   {'PASS' if all(t in text for t in tok) else 'CHECK'}")

# ---------- 7. 结构项 ----------
print("7) 结构要求：")
print(f"   🏃 实战标记：{text.count('🏃')} 处  {'PASS' if text.count('🏃') >= 1 else 'FAIL'}")
n_img = text.count("图片占位")
print(f"   图片占位标注：{n_img} 处  {'PASS' if n_img == 8 else 'CHECK(应为8)'}")
secs = re.findall(r"^## (.+)$", text, re.M)
print(f"   二级标题 {len(secs)} 个：{secs}")
print(f"   首人称「我」出现 {text.count('我')} 次")
print(f"   黑名单「首先/其次/最后」排比：首先={text.count('首先')} 其次={text.count('其次')}")
print(f"   GitHub 链接：{text.count('github.com/beverlyLee')} 处")
print("=" * 62)
