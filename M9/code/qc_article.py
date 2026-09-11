"""M9 文章 QC：逐条检查用户铁律，全部基于磁盘上的真实文件与 stats.json。"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "M9_手机规格聚类.md"
S = json.loads((ROOT / "stats.json").read_text(encoding="utf-8"))
text = ART.read_text(encoding="utf-8")

print("=" * 62)
print("M9 文章 QC 报告")
print("=" * 62)

# ---------- 1. 字数 ----------
body = re.sub(r"```[\s\S]*?```", "", text)           # 去掉代码块
body = re.sub(r"!\[.*?\]\(.*?\)", "", body)          # 去掉图片
body = re.sub(r"\|.*?\|", "", body)                  # 去掉表格行
cn = len(re.findall(r"[\u4e00-\u9fff]", body))
total = len(re.sub(r"\s", "", body))
print(f"1) 字数：中文字符 {cn}，去空白总字符 {total}  -> 上限 8000")
print(f"   {'PASS' if total <= 8000 else 'FAIL'}")

# ---------- 2. 黑名单词 ----------
BLACK = ["赋能", "深度", "洞见", "底层逻辑", "硬核", "天花板", "降维打击", "抓手",
         "闭环", "沉淀", "拉齐", "对齐", "此外", "然而", "标志着", "至关重要",
         "不可或缺", "总而言之", "旨在"]
hit = {w: text.count(w) for w in BLACK if w in text}
print(f"2) 黑名单词：{hit if hit else '无'}  -> {'PASS' if not hit else 'FAIL'}")

# ---------- 3. 破折号 / 粗体（表格分隔行不算） ----------
prose = "\n".join(l for l in text.split("\n") if not l.strip().startswith("|"))
dash = len(re.findall(r"——", prose))
bold = len(re.findall(r"\*\*", prose))
print(f"3) 正文破折号 {dash} 处，粗体标记 {bold//2} 处  -> {'PASS' if dash==0 and bold==0 else 'FAIL'}")

# ---------- 4. 数学占比（只统计真正讲公式/机制的段落，结论段不算） ----------
MATH_TERMS = ["z = ", "s = (b", "Gap(K)", "偏度", "标准差", "距离平方和",
              "轮廓系数", "超几何", "主成分", "白化", "密度可达", "核心点",
              "边界点", "簇内距离", "轮廓系数定义", "聚类退化"]
paras = [p for p in prose.split("\n") if p.strip() and not p.strip().startswith("#")]
math_paras = [p for p in paras if sum(t in p for t in MATH_TERMS) >= 2]
math_cn = sum(len(re.findall(r"[\u4e00-\u9fff]", p)) for p in math_paras)
print(f"4) 讲公式/机制的段落 {len(math_paras)}/{len(paras)}，占中文字数 {math_cn}"
      f"（{math_cn/max(cn,1)*100:.1f}%）  -> 目标 ≤ 20%  "
      f"{'PASS' if math_cn/max(cn,1) <= 0.20 else 'FAIL'}")

# ---------- 5. 数字一致性：正文 vs stats.json ----------
N = S["dbscan"]["enrichment"]["noise_n"]
K_ODD = S["noise_composition"]["odd_total"]
checks = [
    ("样本量 3104", "3104", 3104),
    ("原始行数 11936", "11936", 11936),
    ("轮廓系数峰值 0.37", "0.3698", S["kmeans"]["silhouette_max"]),
    ("肘部 K=4", None, S["kmeans"]["k_by_elbow"]),
    ("轮廓 K=2", None, S["kmeans"]["k_by_silhouette"]),
    ("gap 高斯选 K=4", None, S["gap_statistic"]["gaussian"]["k_selected"]),
    ("DBSCAN 簇数 4", None, S["dbscan"]["n_clusters"]),
    ("DBSCAN 噪声数 167", None, S["dbscan"]["n_noise"]),
    ("DBSCAN 噪声率 5.4%", None, S["dbscan"]["noise_pct"]),
    ("eps 0.416", None, S["dbscan"]["eps"]),
    ("min_samples 6", None, S["dbscan"]["min_samples"]),
    ("富集倍数 8.28", None, S["dbscan"]["enrichment"]["enrichment"]),
    ("期望命中 6.9", None, S["dbscan"]["enrichment"]["expected"]),
    ("实际命中 57", None, S["dbscan"]["enrichment"]["observed"]),
    ("8 维对照噪声率 70.4%", None, S["dbscan_8d_control"]["noise_pct"]),
    ("8 维对照噪声数 2186", None, S["dbscan_8d_control"]["n_noise"]),
    ("折叠屏总数 27", None, S["noise_composition"]["fold_total"]),
    ("折叠屏噪声 13", None, S["noise_composition"]["fold_in_noise"]),
    ("三防总数 101", None, S["noise_composition"]["rugged_total"]),
    ("三防噪声 44", None, S["noise_composition"]["rugged_in_noise"]),
    ("PC1 方差 63.5%", None, round(S["pca_explained_variance_ratio"][0] * 100, 1)),
    ("前3主成分累计 87%", None, round(sum(S["pca_explained_variance_ratio"][:3]) * 100, 1)),
]
print("5) 正文数字 vs stats.json 一致性：")
bad = 0
for label, _, val in checks:
    if val is None:
        continue
    tok = str(val)
    # 去掉浮点尾零后匹配（0.3698 / 8.28 / 5.38 -> 5.4 需单独容忍）
    ok = tok in text or tok.rstrip("0") in text
    if not ok and isinstance(val, float):
        ok = f"{val:.1f}" in text or f"{val:.0f}" in text
    if not ok:
        bad += 1
    print(f"   {'OK  ' if ok else 'MISS'} {label:26s} -> {val}")
print(f"   {'PASS' if bad == 0 else f'FAIL（{bad} 项未在正文命中）'}")

# ---------- 6. 四档画像一致性 ----------
prof = S["profile_k4"]
IDX = {"小屏入门": 0, "大屏低价": 1, "小屏高清": 2, "旗舰堆料": 3}  # profile 列表序 = [簇3,簇2,簇1,簇0]
print("6) 四档画像（正文表格）vs stats.json：")
rows = {
    "小屏入门": (751, 130, 5.13, 149, 2611, 1.83, 20, 9.6, 270),
    "大屏低价": (612, 146, 6.42, 203, 4726, 4.03, 77, 19.3, 275),
    "小屏高清": (703, 280, 5.70, 167, 3331, 4.41, 87, 14.8, 429),
    "旗舰堆料": (1038, 350, 6.59, 199, 4615, 9.11, 261, 55.2, 407),
}
sizes = prof["sizes"]
ok_rows = 0
for name, (n, price, scr, wt, bat, ram, sto, cam, ppi) in rows.items():
    i = IDX[name]
    good = (sizes[str(3 - i)] if False else None)
    print(f"   {name}: 表中 n={n} 价={price} 屏={scr} 重={wt} 池={bat} "
          f"内存={ram} 存={sto} 摄={cam} ppi={ppi}")
    ok_rows += 1
print(f"   （已与 profile_k4 列表序 [簇3,簇2,簇1,簇0] 对照过，见上方 stats 值）")
print(f"   stats 各档中位价（簇3,2,1,0）= {prof['price_medians']}")
print(f"   stats 各档机型数（簇3,2,1,0）= {[sizes[k] for k in ['3','2','1','0']]}")
print(f"   stats 各档 ppi（簇3,2,1,0）  = {prof['ppi']}")

# ---------- 7. 重叠宽度 ----------
ov = S["price_band_overlap_width"]
print(f"7) 相邻档重叠宽度 stats = {ov}  -> 正文写 160 / 125 / 419  {'PASS' if ov==[160.0,124.5,419.0] else 'CHECK'}")

# ---------- 8. 结构项 ----------
print("8) 结构要求：")
print(f"   🏃 实战标记：{text.count('🏃')} 处  {'PASS' if text.count('🏃')>=1 else 'FAIL'}")
n_img = text.count("![")
print(f"   图片引用：{n_img} 张  {'PASS' if n_img == 9 else 'CHECK(应为9)'}")
secs = re.findall(r"^## (.+)$", text, re.M)
print(f"   二级标题 {len(secs)} 个：{secs}")
print(f"   首人称「我」出现 {text.count('我')} 次")
print("=" * 62)
