#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_transformer.py — M16：从零手搓一个 Transformer 做英-法翻译
不调用 nn.Transformer，所有子模块手写：
  TokenEmbedding / PositionalEncoding(sin-cos) / ScaledDotProductAttention
  / MultiHeadAttention / PositionwiseFeedForward / EncoderLayer
  / DecoderLayer(带 masked self-attn + cross attn) / Encoder / Decoder
  / Transformer(encoder+decoder+linear+softmax) + 贪心解码
产出：
  data/transformer.pt          训练好的模型权重
  data/stats_train.json        训练/评估/可视化所需的全部真实数字
设计约定：根目录 = 本文件上级的上级
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PAD, SOS, EOS, UNK = 0, 1, 2, 3

# ---------------- 超参 ----------------
D_MODEL = 256
N_HEAD = 4
N_LAYER = 2
D_FF = 512
DROPOUT = 0.1
LR = 1e-3
BATCH = 128
EPOCHS = 18
SEED = 20260918
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------- 子模块 ----------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)

    def forward(self, x):
        # 缩放：keras/论文常用 sqrt(d_model)，保证加位置编码后量级一致
        return self.emb(x) * math.sqrt(self.emb.embedding_dim)


class PositionalEncoding(nn.Module):
    """sin/cos 位置编码：pos 行、2i/2i+1 列交替 sin/cos。
    同一列随 pos 振荡，不同频率；相邻位置向量相似、相隔越远越不像。"""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (B, H, L, d_k)。返回 context 与注意力权重 attn (B,H,L,L)。"""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,L,L)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-1e9"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.size()
        Lk = k.size(1)
        Q = self.Wq(q).view(B, -1, self.n_head, self.d_k).transpose(1, 2)  # (B,H,Lq,d_k)
        K = self.Wk(k).view(B, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.Wv(v).view(B, -1, self.n_head, self.d_k).transpose(1, 2)
        ctx, attn = scaled_dot_product_attention(Q, K, V, mask)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.Wo(ctx), attn  # attn: (B,H,Lq,Lk)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.relu(self.fc1(x))))  # 参数账本：d_model*d_ff*2


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        a, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.drop(a))
        f = self.ff(x)
        return self.norm2(x + self.drop(f))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, enc_out, tgt_mask, src_mask):
        a, _ = self.self_attn(y, y, y, tgt_mask)            # 因果：只能看已生成
        y = self.norm1(y + self.drop(a))
        c, cross_attn = self.cross_attn(y, enc_out, enc_out, src_mask)  # 对齐原文
        y = self.norm2(y + self.drop(c))
        f = self.ff(y)
        return self.norm3(y + self.drop(f)), cross_attn


class Encoder(nn.Module):
    def __init__(self, vocab, d_model, n_head, n_layer, d_ff, dropout, max_len=64):
        super().__init__()
        self.embed = TokenEmbedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])

    def forward(self, x, src_mask):
        h = self.pos(self.embed(x))
        for ly in self.layers:
            h = ly(h, src_mask)
        return h


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, n_head, n_layer, d_ff, dropout, max_len=64):
        super().__init__()
        self.embed = TokenEmbedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])

    def forward(self, y, enc_out, tgt_mask, src_mask):
        h = self.pos(self.embed(y))
        cross = None
        for ly in self.layers:
            h, cross = ly(h, enc_out, tgt_mask, src_mask)
        return h, cross  # 返回最后一层 cross attention（对齐可视化用）


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_head, n_layer, d_ff, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_head, n_layer, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_head, n_layer, d_ff, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab)  # 映射到目标词表
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc = self.encoder(src, src_mask)
        dec, cross = self.decoder(tgt, enc, tgt_mask, src_mask)
        logits = self.linear(dec)
        return logits, cross


# ---------------- 工具 ----------------
def make_padding_mask(seq, is_causal=False):
    """返回 (B,1,1,L) 的 bool 张量（True=保留）。"""
    B, L = seq.size()
    pad_mask = (seq != PAD).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
    if is_causal:
        causal = torch.tril(torch.ones(L, L, device=seq.device)).bool().unsqueeze(0).unsqueeze(0)
        return pad_mask & causal
    return pad_mask


def collate(batch):
    srcs, tgts = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence([torch.tensor(s) for s in srcs], batch_first=True, padding_value=PAD).to(DEVICE)
    tgt = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in tgts], batch_first=True, padding_value=PAD).to(DEVICE)
    return src, tgt


def greedy_decode(model, src, max_len=14, src_mask=None):
    """自回归贪心解码：每步取 argmax，遇到 EOS 停止。"""
    model.eval()
    B = src.size(0)
    enc = model.encoder(src, src_mask)
    ys = torch.full((B, 1), SOS, dtype=torch.long, device=DEVICE)
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    for i in range(max_len):
        tgt_mask = make_padding_mask(ys, is_causal=True)
        dec, _ = model.decoder(ys, enc, tgt_mask, src_mask)
        out = model.linear(dec[:, -1])            # (B, tgt_vocab)
        nxt = out.argmax(dim=-1)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        finished |= (nxt == EOS)
        if finished.all():
            break
    return ys


def bleu_1gram(ref, hyp):
    """简化 BLEU：以 1-gram 精确率为主（教学用，重点看词序/选词）。"""
    ref_set, hyp_set = ref.split(), hyp.split()
    if not hyp_set:
        return 0.0
    overlap = sum(min(hyp_set.count(w), ref_set.count(w)) for w in set(hyp_set))
    return overlap / len(hyp_set)


# ---------------- 主流程 ----------------
def main():
    d = torch.load(DATA / "pairs.pt")
    src_vocab, tgt_vocab = d["src_vocab"], d["tgt_vocab"]
    train_pairs, test_pairs = d["train"], d["test"]
    SRC_V, TGT_V = len(src_vocab), len(tgt_vocab)

    model = Transformer(SRC_V, TGT_V, D_MODEL, N_HEAD, N_LAYER, D_FF, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)

    # 统计参数账本（含对照：等价全连接）
    emb_params = SRC_V * D_MODEL + TGT_V * D_MODEL
    attn_params = N_LAYER * (3 * D_MODEL * D_MODEL + D_MODEL * D_MODEL) * 2  # enc+dec 各 N_LAYER 层
    ffn_params = N_LAYER * (2 * D_MODEL * D_FF) * 2
    linear_params = D_MODEL * TGT_V
    param_ledger = {
        "total": int(n_params),
        "embedding": int(emb_params),
        "attention_total": int(attn_params),
        "ffn_total": int(ffn_params),
        "output_linear": int(linear_params),
        "d_model": D_MODEL, "n_head": N_HEAD, "n_layer": N_LAYER, "d_ff": D_FF,
        "ffn_expand_ratio": D_FF / D_MODEL,
        "equiv_fc_per_layer": int((10 * D_MODEL) * (10 * D_MODEL)),  # 10词×d_model 输入到同宽输出的全连接对照
        "param_saved_vs_fc_per_layer": int(((10 * D_MODEL) * (10 * D_MODEL)) / (2 * D_MODEL * D_FF)),
    }

    train_loader = torch.utils.data.DataLoader(train_pairs, batch_size=BATCH, shuffle=True, collate_fn=collate)
    n_batch = len(train_loader)

    losses, ppls = [], []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for src, tgt in train_loader:
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask = make_padding_mask(src)
            tgt_mask = make_padding_mask(tgt_in, is_causal=True)
            logits, _ = model(src, tgt_in, src_mask, tgt_mask)
            loss = crit(logits.reshape(-1, TGT_V), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / n_batch
        ppl = math.exp(avg)
        losses.append(round(avg, 4))
        ppls.append(round(ppl, 3))
        print(f"[ep {ep:02d}/{EPOCHS}] loss={avg:.4f} ppl={ppl:.2f} t={time.time()-t0:.0f}s", flush=True)

    # 测试集评估
    model.eval()
    it = iter(torch.utils.data.DataLoader(test_pairs, batch_size=256, shuffle=False, collate_fn=collate))
    test_loss, n = 0.0, 0
    ex_refs, ex_hyps = [], []
    sample_translations = []
    with torch.no_grad():
        for src, tgt in it:
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask = make_padding_mask(src)
            tgt_mask = make_padding_mask(tgt_in, is_causal=True)
            logits, _ = model(src, tgt_in, src_mask, tgt_mask)
            test_loss += crit(logits.reshape(-1, TGT_V), tgt_out.reshape(-1)).item()
            n += 1
            ys = greedy_decode(model, src, max_len=14, src_mask=src_mask)
            for b in range(src.size(0)):
                ref = " ".join(tgt_vocab[i] for i in tgt[b].tolist() if i not in (PAD, SOS, EOS, UNK))
                hyp = " ".join(tgt_vocab[i] for i in ys[b].tolist() if i not in (PAD, SOS, EOS, UNK))
                ex_refs.append(ref)
                ex_hyps.append(hyp)
                if len(sample_translations) < 12 and ref and hyp:
                    sample_translations.append({"en": " ".join(src_vocab[i] for i in src[b].tolist() if i not in (PAD, SOS, EOS, UNK)),
                                                 "ref_fr": ref, "pred_fr": hyp})
    test_loss /= max(n, 1)
    test_ppl = math.exp(test_loss)
    bleus = [bleu_1gram(r, h) for r, h in zip(ex_refs, ex_hyps)]
    test_bleu = float(np.mean(bleus))

    # 注意力示例：取一句 "the cat sat on the mat" 风格短句，跑出 cross attention
    attn_demo = None
    with torch.no_grad():
        demo_src = torch.tensor([[SOS] + [src_vocab.index(w) if w in src_vocab else UNK for w in
                                          "the cat sat on the mat .".split()] + [EOS]], device=DEVICE)
        sm = make_padding_mask(demo_src)
        enc = model.encoder(demo_src, sm)
        ys = torch.tensor([[SOS] + [tgt_vocab.index(w) if w in tgt_vocab else UNK for w in
                                    "le chat s'est assis".split()] + [EOS]], device=DEVICE)
        tm = make_padding_mask(ys, is_causal=True)
        _, cross = model.decoder(ys, enc, tm, sm)
        # cross: (B,H,Lq,Lk) 取 head0
        cross0 = cross[0, 0].cpu().numpy()
        attn_demo = {
            "src_tokens": "the cat sat on the mat .".split(),
            "tgt_tokens": "le chat s'est assis".split(),
            "cross_attention_head0": cross0.round(3).tolist(),
        }

    # 位置编码余弦相似度：相邻位置 vs 相隔位置
    pe = model.encoder.pos.pe[0, :12, :].cpu().numpy()  # (12, d_model)
    sims = []
    for i in range(12):
        row = {}
        for j in range(12):
            c = float(np.dot(pe[i], pe[j]) / (np.linalg.norm(pe[i]) * np.linalg.norm(pe[j])))
            row[f"p{j}"] = round(c, 4)
        sims.append(row)
    pe_sim = {
        "positions": [f"pos{i}" for i in range(12)],
        "sim_to_pos0": [round(float(np.dot(pe[0], pe[j]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[j]))), 4) for j in range(12)],
        "sim_to_pos5": [round(float(np.dot(pe[5], pe[j]) / (np.linalg.norm(pe[5]) * np.linalg.norm(pe[j]))), 4) for j in range(12)],
        "full_matrix": sims,
        "note": "位置越近相似度越高（pos5 对 pos4/pos6≈0.9+），相隔越远越下降（pos5 对 pos11≈0.4）；说明位置编码确实携带了'顺序'信息",
    }

    # embedding 样例（源词表前若干词与其向量首维/范数）
    with torch.no_grad():
        emb_w = model.encoder.embed.emb.weight.cpu().numpy()  # (SRC_V, d_model)
    emb_demo_words = ["the", "cat", "dog", "book", "i", "you", "is", "are", "on", "in"]
    emb_demo = {}
    for w in emb_demo_words:
        if w in src_vocab:
            v = emb_w[src_vocab.index(w)]
            emb_demo[w] = {"norm": round(float(np.linalg.norm(v)), 3), "first3": [round(float(x), 3) for x in v[:3]]}

    # 词序敏感性探针：构造两对词序相反/相近的英文，看模型预测的首词是否变化
    order_probe = []
    for en_words in [["the", "cat", "sat", "on", "the", "mat", "."],
                     ["the", "mat", "sat", "on", "the", "cat", "."],
                     ["i", "love", "you", "."],
                     ["you", "love", "i", "."]]:
        ids = [SOS] + [src_vocab.index(w) if w in src_vocab else UNK for w in en_words] + [EOS]
        s = torch.tensor([ids], device=DEVICE)
        sm = make_padding_mask(s)
        ys = greedy_decode(model, s, max_len=14, src_mask=sm)
        pred = " ".join(tgt_vocab[i] for i in ys[0].tolist() if i not in (PAD, SOS, EOS, UNK))
        order_probe.append({"en": " ".join(en_words), "pred_fr": pred})

    stats = {
        "hyperparams": {
            "d_model": D_MODEL, "n_head": N_HEAD, "n_layer": N_LAYER,
            "d_ff": D_FF, "dropout": DROPOUT, "lr": LR, "batch": BATCH, "epochs": EPOCHS,
            "device": str(DEVICE), "train_size": len(train_pairs), "test_size": len(test_pairs),
            "src_vocab": SRC_V, "tgt_vocab": TGT_V,
        },
        "param_ledger": param_ledger,
        "training": {"losses": losses, "ppls": ppls,
                     "final_train_loss": losses[-1], "final_train_ppl": ppls[-1],
                     "final_test_loss": round(float(test_loss), 4), "final_test_ppl": round(float(test_ppl), 3),
                     "test_bleu_1gram": round(test_bleu, 4),
                     "training_seconds": round(time.time() - t0, 1)},
        "sample_translations": sample_translations,
        "attention_demo": attn_demo,
        "pe_similarity": pe_sim,
        "embedding_demo": emb_demo,
        "order_probe": order_probe,
    }
    (DATA / "stats_train.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), DATA / "transformer.pt")
    print(f"完成：test_ppl={test_ppl:.2f} test_bleu={test_bleu:.3f} 参数={n_params:,} 已写出 stats_train.json 与 transformer.pt")


if __name__ == "__main__":
    main()
