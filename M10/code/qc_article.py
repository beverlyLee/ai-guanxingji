"""QC for M10 article: enforce style, consistency, and truthfulness.

Checks (fail = exit 1):
1. word count (Chinese chars) <= 8000
2. first-person pronoun count >= 1
3. blacklisted AI-flavor words = 0
4. em dash / en dash / Chinese dash count = 0
5. bold emphasis count = 0  (markdown ** or __)
6. math length ratio <= 20%
7. headings count >= 3 and figure refs count >= 5
8. key numbers in text match stats.json
"""
import json, os, re, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTICLE = os.path.join(BASE, "M10_房价决策树.md")
STATS = os.path.join(BASE, "stats.json")

BLACKLIST = [
    "赋能", "深度", "洞见", "底层逻辑", "硬核", "天花板", "降维打击",
    "抓手", "闭环", "沉淀", "拉齐", "对齐", "此外", "然而", "标志着",
    "至关重要", "不可或缺", "总而言之", "旨在"
]

errors = []

text = open(ARTICLE, encoding="utf-8").read()
stats = json.load(open(STATS, encoding="utf-8"))

# 1. word count: Chinese chars + alphanumerics as proxy
chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
words = chinese_chars + len(re.findall(r"[a-zA-Z0-9]+", text))
print(f"word count (Chinese + tokens): {words}")
if words > 8000:
    errors.append(f"word count {words} > 8000")

# 2. first person
first_person = len(re.findall(r"我", text))
print(f"first-person '我' count: {first_person}")
if first_person < 1:
    errors.append("no first-person pronoun found")

# 3. blacklist
black_hits = []
for w in BLACKLIST:
    c = text.count(w)
    if c:
        black_hits.append(f"{w}: {c}")
print(f"blacklist hits: {len(black_hits)}")
if black_hits:
    errors.append("blacklist words: " + ", ".join(black_hits))

# 4. dashes
# em dash, en dash, chinese dash, minus-looking long dash
dash_count = len(re.findall(r"[—–—―]", text))
print(f"dash count: {dash_count}")
if dash_count:
    errors.append(f"found {dash_count} dash characters")

# 5. bold emphasis
bold_count = len(re.findall(r"\*\*[^*]+\*\*|__[^_]+__", text))
print(f"bold emphasis count: {bold_count}")
if bold_count:
    errors.append(f"found {bold_count} bold emphasis markers")

# 6. math ratio
math_blocks = re.findall(r"\$\$.*?\$\$|\$[^$\n]+\$", text, re.DOTALL)
math_len = sum(len(b) for b in math_blocks)
math_ratio = math_len / len(text)
print(f"math ratio: {math_ratio:.3%} ({math_len}/{len(text)})")
if math_ratio > 0.20:
    errors.append(f"math ratio {math_ratio:.1%} > 20%")

# 7. headings and figures
heading_count = len(re.findall(r"^#{1,3} ", text, re.MULTILINE))
fig_count = len(re.findall(r"!\[", text))
print(f"headings: {heading_count}, figure refs: {fig_count}")
if heading_count < 3:
    errors.append(f"headings {heading_count} < 3")
if fig_count < 5:
    errors.append(f"figure refs {fig_count} < 5")

# 8. number consistency with stats.json
checks = [
    (f"median price {stats['median_price']/1e4:.1f} 万", f"{stats['median_price']/1e4:.1f}"),
    (f"n_rows {stats['n_rows']}", str(stats["n_rows"])),
    (f"above_median {stats['above_median']}", str(stats["above_median"])),
    (f"below_median {stats['below_median']}", str(stats["below_median"])),
    (f"root gain {stats['scratch_tree_depth3']['root_gain']}", f"{stats['scratch_tree_depth3']['root_gain']:.4f}"),
    (f"root gain_ratio {stats['scratch_tree_depth3']['root_gr']}", f"{stats['scratch_tree_depth3']['root_gr']:.4f}"),
    (f"entropy root threshold {stats['clf_entropy']['root_thr']}", f"{stats['clf_entropy']['root_thr']:.1f}"),
    (f"entropy root e_left {stats['clf_entropy']['root_e_left']}", f"{stats['clf_entropy']['root_e_left']:.4f}"),
    (f"entropy root e_right {stats['clf_entropy']['root_e_right']}", f"{stats['clf_entropy']['root_e_right']:.4f}"),
    (f"entropy train acc {stats['clf_entropy']['train_acc']}", f"{stats['clf_entropy']['train_acc']:.1%}".replace("%", "")),
    (f"entropy test acc {stats['clf_entropy']['test_acc']}", f"{stats['clf_entropy']['test_acc']:.1%}".replace("%", "")),
    (f"gini train acc {stats['clf_gini']['train_acc']}", f"{stats['clf_gini']['train_acc']:.1%}".replace("%", "")),
    (f"gini test acc {stats['clf_gini']['test_acc']}", f"{stats['clf_gini']['test_acc']:.1%}".replace("%", "")),
    (f"deep reg train r2 {stats['reg_deep']['train_r2']}", f"{stats['reg_deep']['train_r2']:.3f}"),
    (f"deep reg test r2 {stats['reg_deep']['test_r2']}", f"{stats['reg_deep']['test_r2']:.3f}"),
    (f"deep reg leaves {stats['reg_deep']['leaves']}", str(stats["reg_deep"]["leaves"])),
    (f"shallow d3 train r2 {stats['reg_shallow_d3']['train_r2']}", f"{stats['reg_shallow_d3']['train_r2']:.3f}"),
    (f"shallow d3 test r2 {stats['reg_shallow_d3']['test_r2']}", f"{stats['reg_shallow_d3']['test_r2']:.3f}"),
    (f"pre-prune best depth {stats['pre_prune_best_depth']}", str(stats["pre_prune_best_depth"])),
    (f"pre-prune best test r2 {stats['pre_prune']['test_r2'][stats['pre_prune_best_depth']-1]}", f"{stats['pre_prune']['test_r2'][stats['pre_prune_best_depth']-1]:.3f}"),
    (f"post-prune alpha {stats['post_prune']['alpha']}", f"{stats['post_prune']['alpha']:.1e}"),
    (f"post-prune test r2 {stats['post_prune']['test_r2']}", f"{stats['post_prune']['test_r2']:.3f}"),
    (f"post-prune leaves {stats['post_prune']['leaves']}", str(stats["post_prune"]["leaves"])),
    (f"boundary test r2 {stats['boundary']['test_r2']}", f"{stats['boundary']['test_r2']:.3f}"),
]

missing_numbers = []
for label, needle in checks:
    # be tolerant of formatting variants: e.g. 16.3 vs 16.30
    if needle not in text and needle.rstrip("0").rstrip(".") not in text:
        # also try percent form
        if not (needle.endswith(".") and needle[:-1] in text):
            missing_numbers.append(label)

print(f"numbers missing from article: {len(missing_numbers)}")
if missing_numbers:
    errors.append("missing numbers: " + ", ".join(missing_numbers[:10]))

# Title check
title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
if title_match:
    title = title_match.group(1)
    print(f"title: {title}")
    if "决策树" not in title:
        errors.append("title does not contain 决策树")
else:
    errors.append("no H1 title found")

if errors:
    print("\nQC FAILED:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("\nQC PASSED")
