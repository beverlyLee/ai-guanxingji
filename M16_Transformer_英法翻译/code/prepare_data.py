#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py — M16 Transformer 英法翻译：数据准备
数据集：Tatoeba 英-法平行句对（OPUS 镜像，权威且零政治敏感）
产出：
  data/Tatoeba.en-fr.en / .fr （解压后）
  data/vocab_src.json, data/vocab_tgt.json  （词表：index->word）
  data/pairs.pt  （train/test 整数序列）
  data/stats.json  （数据规模 + 词表规模，文章所有数字的唯一来源之一）
设计约定：根目录 = 本文件上级的上级（M16_Transformer_英法翻译/）
"""
import json
import re
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# 下载源（按优先级）：主源 OPUS Tatoeba；其余为已知历史镜像（可能 403/404，仅作兜底）
URLS = [
    "https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/moses/en-fr.txt.zip",
    "https://www.manythings.org/anki/eng-fra.zip",
    "https://download.pytorch.org/tutorial/data/eng-fra.zip",
]

MAX_LEN = 10       # 每侧句子最多 10 个词（短句更适合讲清词序与注意力）
MIN_LEN = 1
MAX_VOCAB = 8000   # 词表上限，超出则用 UNK（词表仍从全量过滤语料统计，保证低频词少）
TEST_RATIO = 0.10
TRAIN_CAP = 60000  # 训练实际使用的句对数（全量 21.9 万对，这里抽样以保证可复现 + 训练可在分钟级完成）
TEST_CAP = 6000    # 测试实际使用的句对数
SEED = 20260918

# 特殊 token
PAD, SOS, EOS, UNK = 0, 1, 2, 3

# 允许字符：字母（任意语言，含重音）、空格、基础标点；数字与其他符号一律丢弃
_ALLOWED = re.compile(r"^[A-Za-zÀ-ÿĀ-žЀ-ӿ\s.,!?;:'\"(){}\-]+$")


def log(msg: str):
    print(f"[prepare] {msg}", flush=True)


def download():
    zip_path = DATA / "en-fr.txt.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        log(f"已存在压缩包 {zip_path.name}（{zip_path.stat().st_size/1e6:.1f} MB），跳过下载")
        return zip_path
    for url in URLS:
        try:
            log(f"尝试下载：{url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(zip_path, "wb") as f:
                f.write(r.read())
            if zip_path.stat().st_size > 1_000_000:
                log(f"下载成功：{zip_path.name}（{zip_path.stat().st_size/1e6:.1f} MB）")
                return zip_path
        except Exception as e:  # noqa: BLE001
            log(f"  失败：{e}")
    raise RuntimeError("所有下载源均不可用")


def unzip_pairs(zip_path: Path):
    en_file = DATA / "Tatoeba.en-fr.en"
    fr_file = DATA / "Tatoeba.en-fr.fr"
    if en_file.exists() and fr_file.exists():
        return en_file, fr_file
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # 优先找 Tatoeba.en-fr.en / .fr；很多镜像是 eng-fra.txt（tab 分隔）
        en = next((n for n in names if n.endswith(".en")), None)
        fr = next((n for n in names if n.endswith(".fr")), None)
        txt = next((n for n in names if n.endswith(".txt")), None)
        if en and fr:
            z.extract(en, DATA)
            z.extract(fr, DATA)
            return DATA / en, DATA / fr
        if txt:
            z.extract(txt, DATA)
            p = DATA / txt
            out_en, out_fr = DATA / "Tatoeba.en-fr.en", DATA / "Tatoeba.en-fr.fr"
            with open(p, encoding="utf-8") as fh:
                lines = [ln.rstrip("\n").split("\t") for ln in fh if "\t" in ln]
            with open(out_en, "w", encoding="utf-8") as fe, open(out_fr, "w", encoding="utf-8") as ff:
                for parts in lines:
                    if len(parts) >= 2:
                        fe.write(parts[0] + "\n")
                        ff.write(parts[1] + "\n")
            return out_en, out_fr
    raise RuntimeError("压缩包内未找到可用的英-法对齐文件")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def keep(s: str) -> bool:
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    if not _ALLOWED.match(s):
        return False
    return True


def tokenize(s: str):
    return s.split(" ")


def build_vocab(sents, max_vocab):
    from collections import Counter
    cnt = Counter()
    for s in sents:
        cnt.update(s)
    vocab = ["<pad>", "<sos>", "<eos>", "<unk>"]
    for w, _ in cnt.most_common(max_vocab - len(vocab)):
        vocab.append(w)
    return vocab


def main():
    zip_path = download()
    en_file, fr_file = unzip_pairs(zip_path)
    with open(en_file, encoding="utf-8") as f:
        ens = f.read().splitlines()
    with open(fr_file, encoding="utf-8") as f:
        frs = f.read().splitlines()
    pairs_raw = min(len(ens), len(frs))
    log(f"原始对齐句对：{pairs_raw}")

    pairs = []
    dropped_long = 0
    dropped_bad = 0
    for e, f in zip(ens, frs):
        e_n, f_n = normalize(e), normalize(f)
        if not (keep(e_n) and keep(f_n)):
            dropped_bad += 1
            continue
        et, ft = tokenize(e_n), tokenize(f_n)
        if not (MIN_LEN <= len(et) <= MAX_LEN and MIN_LEN <= len(ft) <= MAX_LEN):
            dropped_long += 1
            continue
        pairs.append((et, ft))
    log(f"过滤后可用句对：{len(pairs)}（过长丢弃 {dropped_long}，含数字/特殊字符丢弃 {dropped_bad}）")

    # 词表
    src_vocab = build_vocab([p[0] for p in pairs], MAX_VOCAB)
    tgt_vocab = build_vocab([p[1] for p in pairs], MAX_VOCAB)
    src2i = {w: i for i, w in enumerate(src_vocab)}
    tgt2i = {w: i for i, w in enumerate(tgt_vocab)}
    log(f"源词表 {len(src_vocab)}，目标词表 {len(tgt_vocab)}")

    def encode(sent, s2i):
        return [SOS] + [s2i.get(w, UNK) for w in sent] + [EOS]

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pairs))
    n_test = int(len(pairs) * TEST_RATIO)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    # 抽样到可训练规模（词表仍基于全量统计，UNK 率保持低位）
    train_idx = train_idx[:TRAIN_CAP]
    test_idx = test_idx[:TEST_CAP]

    train_pairs = [(encode(pairs[i][0], src2i), encode(pairs[i][1], tgt2i)) for i in train_idx]
    test_pairs = [(encode(pairs[i][0], src2i), encode(pairs[i][1], tgt2i)) for i in test_idx]

    # 保存
    (DATA / "vocab_src.json").write_text(json.dumps(src_vocab, ensure_ascii=False), encoding="utf-8")
    (DATA / "vocab_tgt.json").write_text(json.dumps(tgt_vocab, ensure_ascii=False), encoding="utf-8")
    import torch
    torch.save(
        {
            "train": train_pairs,
            "test": test_pairs,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "special": {"PAD": PAD, "SOS": SOS, "EOS": EOS, "UNK": UNK},
        },
        DATA / "pairs.pt",
    )

    src_lens = [len(p[0]) for p in pairs]
    tgt_lens = [len(p[1]) for p in pairs]
    stats = {
        "dataset": {
            "name": "Tatoeba 英-法平行句对 (OPUS 镜像 v20190709)",
            "url": URLS[0],
            "pairs_raw": int(pairs_raw),
            "pairs_used": int(len(pairs)),
            "dropped_too_long": int(dropped_long),
            "dropped_bad_chars": int(dropped_bad),
            "max_len_per_side": MAX_LEN,
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
            "train_size": len(train_pairs),
            "test_size": len(test_pairs),
            "train_cap": TRAIN_CAP,
            "test_cap": TEST_CAP,
            "note": "训练/测试仅使用过滤后语料的抽样子集；词表由全量过滤语料统计得到",
            "src_len_mean": round(float(np.mean(src_lens)), 3),
            "src_len_max": int(max(src_lens)),
            "tgt_len_mean": round(float(np.mean(tgt_lens)), 3),
            "tgt_len_max": int(max(tgt_lens)),
        },
        "sample_pairs": [{"en": " ".join(p[0]), "fr": " ".join(p[1])} for p in pairs[:8]],
    }
    (DATA / "stats_data.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写出 vocab_src.json / vocab_tgt.json / pairs.pt / stats_data.json")
    log("数据准备完成。")


if __name__ == "__main__":
    main()
