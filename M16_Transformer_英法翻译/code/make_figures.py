#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — M16：从 stats_train.json + transformer.pt 重生 8 张配图
风格：scientific-visualization（STHeiti 中文、Okabe-Ito 色盲安全、去脊线、无 chart junk）
根目录 = 本文件上级的上级；PNG 写到 figures/
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------- 字体与样式 ----------------
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
if Path(FONT_PATH).exists():
    fm.fontManager.addfont(FONT_PATH)
    CN = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = CN.get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Okabe-Ito 色盲安全配色
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
    "gray": "#999999",
}
plt.rcParams["axes.edgecolor"] = "#333333"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name)


# ---------------- 1. 整体架构示意图 ----------------
def fig_architecture(stats):
    d = stats["hyperparams"]
    nL = d["n_layer"]
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    enc_x, enc_w = 0.07, 0.30
    dec_x, dec_w = 0.63, 0.30

    def box(x, y, w, h, text, fc, tc="#000000", fs=12, lw=1.6):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#333333", lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontproperties=CN, fontsize=fs, color=tc, wrap=True)

    # 编码器（自下而上堆叠）
    enc_modules = ["输入嵌入 ×√d\n+ 位置编码(sin/cos)", "多头自注意力", "前馈 FFN\n(d→4d→d)"]
    base_y = 0.12
    h = 0.16
    gap = 0.04
    ey = base_y
    for m in enc_modules:
        box(enc_x, ey, enc_w, h, m, OI["sky"], fs=11)
        ey += h + gap
    # 整体编码器外框 ×nL
    enc_top = ey - gap
    ax.add_patch(plt.Rectangle((enc_x - 0.012, base_y - 0.012), enc_w + 0.024,
                               (enc_top - base_y) + 0.024, fc="none", ec=OI["blue"], lw=2.2))
    ax.text(enc_x + enc_w / 2, enc_top + 0.02, f"编码器 ×{nL}\n(自注意力 + 前馈)",
            ha="center", va="bottom", fontproperties=CN, fontsize=12, color=OI["blue"], weight="bold")

    # 解码器
    dec_modules = ["输出嵌入 ×√d\n+ 位置编码", "掩码多头自注意力\n(因果: 只看已生成)",
                  "交叉注意力\n(K,V=编码器输出)", "前馈 FFN\n(d→4d→d)"]
    dy = base_y
    for m in dec_modules:
        box(dec_x, dy, dec_w, h, m, OI["green"], fs=10.5)
        dy += h + gap
    dec_top = dy - gap
    ax.add_patch(plt.Rectangle((dec_x - 0.012, base_y - 0.012), dec_w + 0.024,
                               (dec_top - base_y) + 0.024, fc="none", ec=OI["green"], lw=2.2))
    ax.text(dec_x + dec_w / 2, dec_top + 0.02, f"解码器 ×{nL}\n(自注意力 + 交叉注意力 + 前馈)",
            ha="center", va="bottom", fontproperties=CN, fontsize=12, color=OI["green"], weight="bold")

    # 顶部：线性 + softmax
    box(dec_x - 0.012, dec_top + 0.10, dec_w + 0.024, 0.12,
        "线性层 Linear → Softmax → 词表概率分布", OI["orange"], fs=11)
    ax.annotate("", xy=(dec_x + dec_w / 2, dec_top + 0.10), xytext=(dec_x + dec_w / 2, dec_top + 0.025),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))

    # 编码器 → 解码器（交叉注意力来源）
    ax.annotate("", xy=(dec_x, (base_y + dec_top) / 2 + 0.05), xytext=(enc_x + enc_w, (base_y + enc_top) / 2 + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=OI["verm"], lw=2.0))
    ax.text((enc_x + enc_w + dec_x) / 2, (base_y + enc_top) / 2 + 0.08,
            "编码向量 K,V（原文语义）", ha="center", fontproperties=CN, fontsize=10, color=OI["verm"])

    # 自回归解码说明
    ax.text(0.5, 0.02, "自回归生成：每步把已生成词喂回解码器，贪心取 argmax，遇到 EOS 停止",
            ha="center", fontproperties=CN, fontsize=10.5, color="#333333")
    ax.text(0.5, 0.97, "Transformer 整体架构（从零手搓，英 → 法 翻译）",
            ha="center", fontproperties=CN, fontsize=14, weight="bold", color="#222222")
    _save(fig, "fig_architecture.png")


# ---------------- 2. 分词与词 ID 示例 ----------------
def fig_tokenization(stats, vocab_src):
    # vocab_src 是 list[id]->token；先建 token->id 反查表
    if isinstance(vocab_src, list):
        w2i = {w: i for i, w in enumerate(vocab_src)}
    else:
        w2i = vocab_src
    # 选用全部在词表内的示例句，避免 UNK 干扰；to 重复出现 -> 同一 ID（查表性质）
    demo = "he wanted to talk to tom".split()
    ids = [w2i.get(w, 3) for w in demo]  # 3=UNK
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, len(demo)); ax.set_ylim(0, 1); ax.axis("off")
    colors = [OI["sky"], OI["orange"], OI["green"], OI["purple"], OI["verm"], OI["blue"], OI["yellow"]]
    for i, (w, idx) in enumerate(zip(demo, ids)):
        c = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle((i + 0.05, 0.45), 0.9, 0.42, fc=c, ec="#333333", lw=1.4))
        ax.text(i + 0.5, 0.66, w, ha="center", va="center", fontproperties=CN, fontsize=13, color="#000")
        ax.text(i + 0.5, 0.30, f"id={idx}", ha="center", va="center", fontproperties=CN, fontsize=11, color="#333")
    ax.text(len(demo) / 2, 0.97, "Tokenization：句子先被切成词（token），再映射到词表中的整数 ID",
            ha="center", fontproperties=CN, fontsize=12, weight="bold")
    _save(fig, "fig_tokenization.png")


# ---------------- 3. 位置编码余弦相似度热力图 ----------------
def fig_pe_similarity(stats):
    pe = stats["pe_similarity"]
    fm = pe["full_matrix"]
    n = len(fm)
    # full_matrix[i] 是 {p0..p11: 相似度} 的字典，组装成方阵
    M = np.array([[float(fm[i][f"p{j}"]) for j in range(n)] for i in range(n)], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(M, cmap="viridis", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"p{i}" for i in range(n)], fontproperties=CN, fontsize=10)
    ax.set_yticklabels([f"p{i}" for i in range(n)], fontproperties=CN, fontsize=10)
    ax.set_xlabel("位置", fontproperties=CN, fontsize=11)
    ax.set_ylabel("位置", fontproperties=CN, fontsize=11)
    ax.set_title("位置编码余弦相似度：相邻位置高（≈0.97）、越远越降（≈0.66）", fontproperties=CN, fontsize=11.5, weight="bold")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] < 0.75 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig_pe_similarity.png")


# ---------------- 4. 交叉注意力热力图 ----------------
def fig_cross_attention(stats):
    ad = stats["attention_demo"]
    mat = np.array(ad["cross_attention_head0"], dtype=float)
    # 矩阵为 (tgt_len x src_len)，其中 tgt/src 都含 <sos>/<eos>：
    # 源 9 列 = <sos> + 7 词 + <eos>，目标 6 行 = <sos> + 4 词 + <eos>
    src = ["<sos>"] + ad["src_tokens"] + ["<eos>"]
    tgt = ["<sos>"] + ad["tgt_tokens"] + ["<eos>"]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    im = ax.imshow(mat, cmap="OrRd")
    ax.set_xticks(range(len(src))); ax.set_yticks(range(len(tgt)))
    ax.set_xticklabels(src, fontproperties=CN, fontsize=10, rotation=35, ha="right")
    ax.set_yticklabels(tgt, fontproperties=CN, fontsize=10)
    ax.set_xlabel("英文（编码器输出 K,V）", fontproperties=CN, fontsize=11)
    ax.set_ylabel("法文（解码器查询 Q）", fontproperties=CN, fontsize=11)
    ax.set_title("交叉注意力 head0：法文每个词在‘看’哪个英文词", fontproperties=CN, fontsize=12, weight="bold")
    for i in range(len(tgt)):
        for j in range(len(src)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig_cross_attention.png")


# ---------------- 5. 多头注意力示意 ----------------
def fig_multihead(stats):
    d = stats["hyperparams"]
    nH = d["n_head"]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # 输入
    ax.add_patch(plt.Rectangle((0.02, 0.35), 0.12, 0.30, fc=OI["sky"], ec="#333"))
    ax.text(0.08, 0.5, "输入\n序列", ha="center", va="center", fontproperties=CN, fontsize=10)
    # 切 4 个头
    head_colors = [OI["orange"], OI["green"], OI["purple"], OI["verm"]]
    hx = 0.20
    hw = 0.13
    for i in range(nH):
        ax.add_patch(plt.Rectangle((hx + i * (hw + 0.02), 0.35), hw, 0.30, fc=head_colors[i], ec="#333"))
        ax.text(hx + i * (hw + 0.02) + hw / 2, 0.5, f"头{i+1}\n注意力", ha="center", va="center",
                fontproperties=CN, fontsize=9.5, color="#000")
    # 拼接
    ax.add_patch(plt.Rectangle((0.80, 0.35), 0.10, 0.30, fc=OI["yellow"], ec="#333"))
    ax.text(0.85, 0.5, "拼接\n+Wo", ha="center", va="center", fontproperties=CN, fontsize=9.5)
    # 箭头
    for i in range(nH):
        ax.annotate("", xy=(0.80, 0.5), xytext=(hx + i * (hw + 0.02) + hw, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    ax.annotate("", xy=(0.20, 0.5), xytext=(0.14, 0.5), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    ax.text(0.5, 0.92, f"多头自注意力：把 d={d['d_model']} 切成 {nH} 个并行的 {d['d_model']//nH} 维子空间，各看不同的词关系",
            ha="center", fontproperties=CN, fontsize=11.5, weight="bold")
    ax.text(0.5, 0.10, "4 个头分别捕捉不同模式（如语法/指代/距离），最后拼接再投影，比单头看得全",
            ha="center", fontproperties=CN, fontsize=10, color="#333")
    _save(fig, "fig_multihead.png")


# ---------------- 6. FFN 维度账本 ----------------
def fig_ffn_ledger(stats):
    pl = stats["param_ledger"]
    dm, df = pl["d_model"], pl["d_ff"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _clean(ax)
    cats = ["嵌入层\n(源+目标)", "注意力\n(全部层)", "前馈 FFN\n(全部层)", "输出线性\n→词表"]
    vals = [pl["embedding"] / 1e6, pl["attention_total"] / 1e6, pl["ffn_total"] / 1e6, pl["output_linear"] / 1e6]
    colors = [OI["sky"], OI["orange"], OI["green"], OI["verm"]]
    bars = ax.bar(cats, vals, color=colors, edgecolor="#333", lw=1.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}M", ha="center",
                fontproperties=CN, fontsize=11, weight="bold")
    ax.set_ylabel("参数量（百万）", fontproperties=CN, fontsize=11)
    ax.set_title(f"参数账本：总 {pl['total']/1e6:.2f}M，FFN 把 {dm} 维先扩到 {df} 维再压回（{pl['ffn_expand_ratio']:.0f}× 扩展）",
                 fontproperties=CN, fontsize=11.5, weight="bold")
    _save(fig, "fig_ffn_ledger.png")


# ---------------- 7. 训练 loss / ppl 曲线 ----------------
def fig_loss_curve(stats):
    tr = stats["training"]
    eps = list(range(1, len(tr["losses"]) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    _clean(ax1)
    ax1.plot(eps, tr["losses"], color=OI["blue"], marker="o", ms=4, lw=2, label="训练 loss")
    ax1.set_xlabel("轮次 epoch", fontproperties=CN, fontsize=11)
    ax1.set_ylabel("交叉熵 loss", fontproperties=CN, fontsize=11, color=OI["blue"])
    ax1.tick_params(axis="y", labelcolor=OI["blue"])
    ax2 = ax1.twinx()
    ax2.plot(eps, tr["ppls"], color=OI["verm"], marker="s", ms=4, lw=2, label="困惑度 PPL")
    ax2.set_ylabel("困惑度 PPL = exp(loss)", fontproperties=CN, fontsize=11, color=OI["verm"])
    ax2.tick_params(axis="y", labelcolor=OI["verm"])
    ax1.set_title(f"训练曲线：最终 test_ppl={tr['final_test_ppl']}，test_bleu(1-gram)={tr['test_bleu_1gram']}",
                  fontproperties=CN, fontsize=11.5, weight="bold")
    _save(fig, "fig_loss_curve.png")


# ---------------- 8. softmax 输出分布 ----------------
def build_softmax_fig(stats, src_vocab, tgt_vocab):
    import train_transformer as T  # 复用模型定义
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # vocab 为 list[id]->token，建反查表
    if isinstance(src_vocab, list):
        src_w2i = {w: i for i, w in enumerate(src_vocab)}
    else:
        src_w2i = src_vocab
    if isinstance(tgt_vocab, list):
        tgt_w2i = {w: i for i, w in enumerate(tgt_vocab)}
    else:
        tgt_w2i = tgt_vocab
    d = stats["hyperparams"]
    model = T.Transformer(len(src_vocab), len(tgt_vocab), d["d_model"], d["n_head"],
                          d["n_layer"], d["d_ff"], d["dropout"]).to(device)
    model.load_state_dict(torch.load(T.DATA / "transformer.pt", map_location=device))
    model.eval()
    # 取文章示例英文句，看首词概率分布（the/cat/sat/on/the/mat/. 中 mat 与 . 为 UNK，但首词预测不受影响）
    en = "the cat sat on the mat .".split()
    ids = [T.SOS] + [src_w2i.get(w, T.UNK) for w in en] + [T.EOS]
    src = torch.tensor([ids], device=device)
    src_mask = T.make_padding_mask(src)
    enc = model.encoder(src, src_mask)
    ys = torch.tensor([[T.SOS]], device=device)
    tm = T.make_padding_mask(ys, is_causal=True)
    dec, _ = model.decoder(ys, enc, tm, src_mask)
    logits = model.linear(dec[:, -1])
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    topk = np.argsort(probs)[::-1][:10]
    labels = [tgt_vocab[i] for i in topk]
    vals = probs[topk]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _clean(ax)
    colors = [OI["verm"] if i == 0 else OI["blue"] for i in range(len(labels))]
    bars = ax.barh(range(len(labels))[::-1], vals, color=colors, edgecolor="#333", lw=1.1)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels([f"{l}" for l in labels], fontproperties=CN, fontsize=11)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center",
                fontproperties=CN, fontsize=9.5)
    ax.set_xlabel("概率", fontproperties=CN, fontsize=11)
    ax.set_title("Linear+Softmax 输出：法文首词的概率分布（模型最看好 le）", fontproperties=CN,
                 fontsize=11.5, weight="bold")
    _save(fig, "fig_softmax.png")


def main():
    stats = json.loads((DATA / "stats_train.json").read_text(encoding="utf-8"))
    vocab_src = json.loads((DATA / "vocab_src.json").read_text(encoding="utf-8"))
    vocab_tgt = json.loads((DATA / "vocab_tgt.json").read_text(encoding="utf-8"))

    fig_architecture(stats)
    fig_tokenization(stats, vocab_src)
    fig_pe_similarity(stats)
    fig_cross_attention(stats)
    fig_multihead(stats)
    fig_ffn_ledger(stats)
    fig_loss_curve(stats)
    build_softmax_fig(stats, vocab_src, vocab_tgt)
    print("全部配图已生成于 figures/")


if __name__ == "__main__":
    main()
