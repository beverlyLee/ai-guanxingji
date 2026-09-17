#!/usr/bin/env python3
"""错例分析：把 43 类按语义归成 4 个家族，看误判落在族内还是跨族。

输入：data/plot_data.npz（混淆矩阵）、data/stats.json（类名）
输出：data/error_analysis.json —— 文章里所有错例数字的唯一来源

家族划分（四种交通标志语义）：
    限速族  0-8, 32            限速 20~120、解除限速、解除全部限制
    禁令族  9,10,13,14,15,16,17,41,42  禁止超车/禁行/让行/停车/解除禁止
    指示族  33-40              必须直行/左转/右转/靠左右/环岛
    警告族  11,12,18-31        注意危险、急弯、湿滑、施工、行人等
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FAMILIES = {
    "限速族": [0, 1, 2, 3, 4, 5, 6, 7, 8, 32],
    "禁令族": [9, 10, 13, 14, 15, 16, 17, 41, 42],
    "指示族": [33, 34, 35, 36, 37, 38, 39, 40],
    "警告族": [11, 12, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
}
FAM_OF = {c: f for f, cs in FAMILIES.items() for c in cs}


def analyse(cm, names):
    n = cm.shape[0]
    assert len(FAM_OF) == n, f"家族覆盖 {len(FAM_OF)} 类，却有 {n} 类"
    wrong = int(cm.sum() - np.trace(cm))

    # 错例按家族：族内（真实、预测同族）vs 跨族
    inner = {f: 0 for f in FAMILIES}
    cross = 0
    for t in range(n):
        for p in range(n):
            if t == p or cm[t, p] == 0:
                continue
            if FAM_OF[t] == FAM_OF[p]:
                inner[FAM_OF[t]] += int(cm[t, p])
            else:
                cross += int(cm[t, p])

    # 最易混的对（真实类被判错的次数，降序）
    pairs = []
    for t in range(n):
        for p in range(n):
            if t != p and cm[t, p] > 0:
                pairs.append({"true": t, "pred": p, "count": int(cm[t, p]),
                              "true_name": names[t], "pred_name": names[p],
                              "same_family": FAM_OF[t] == FAM_OF[p]})
    pairs.sort(key=lambda d: (-d["count"], d["true"]))

    # 预测分布：模型把测试图都判成了谁（只看判错的那部分）
    pred_of_wrong = (cm.sum(axis=0) - np.diag(cm)).astype(int)
    top_pred = np.argsort(pred_of_wrong)[::-1][:5]

    # 每个真实类被抓错多少张（按错得多的排在前面）
    per_cls = []
    for t in range(n):
        w = int(cm[t, :].sum() - cm[t, t])
        if w:
            per_cls.append({"cls": t, "name": names[t], "wrong": w,
                            "n_test": int(cm[t, :].sum()),
                            "recall": float(cm[t, t] / max(cm[t, :].sum(), 1))})
    per_cls.sort(key=lambda d: (-d["wrong"], d["cls"]))

    # 每个真实家族的错例数占该家族样本数的比例
    fam_stat = {}
    for f, cs in FAMILIES.items():
        tot = int(cm[cs, :].sum())
        w = int(tot - cm[np.ix_(cs, cs)].trace())
        fam_stat[f] = {"n": tot, "wrong": w,
                       "acc": (tot - w) / tot if tot else 0.0}

    return {
        "wrong_total": wrong,
        "inner": inner,
        "cross": cross,
        "pairs": pairs[:12],
        "wrong_classes": per_cls,
        "top_pred_correct": int(np.diag(cm)[int(np.argmax(np.diag(cm)))]),
        "pred_of_wrong_top": [
            {"cls": int(c), "name": names[int(c)], "count": int(pred_of_wrong[c]),
             "share": float(pred_of_wrong[c] / max(wrong, 1))} for c in top_pred],
        "family": fam_stat,
    }


def split_stat(names):
    """切分后每一类真正拿到多少张：原始 GTSRB 差 10 倍，取用后缩到多少。"""
    d = np.load(DATA / "gtsrb32.npz")
    ytr, yte = d["y_train"], d["y_test"]
    ctr = np.bincount(ytr, minlength=len(names))
    cte = np.bincount(yte, minlength=len(names))
    return {
        "train_per_class_min": int(ctr.min()), "train_per_class_max": int(ctr.max()),
        "train_imbalance": float(ctr.max() / ctr.min()),
        "test_per_class_min": int(cte.min()), "test_per_class_max": int(cte.max()),
        "test_imbalance": float(cte.max() / cte.min()),
        "train_smallest": names[int(np.argmin(ctr))],
        "train_largest": names[int(np.argmax(ctr))],
    }


def main():
    st = json.loads((DATA / "stats.json").read_text(encoding="utf-8"))
    npz = np.load(DATA / "plot_data.npz")
    names = st["class_names"]

    out = {"family_map": {k: [names[c] for c in v] for k, v in FAMILIES.items()}}
    out["split"] = split_stat(names)
    print("=== 切分后每类样本数 ===")
    print("  训练", out["split"]["train_per_class_min"], "~",
          out["split"]["train_per_class_max"], "差",
          f"{out['split']['train_imbalance']:.2f} 倍")
    print("  测试", out["split"]["test_per_class_min"], "~",
          out["split"]["test_per_class_max"], "差",
          f"{out['split']['test_imbalance']:.2f} 倍")
    for r in st["runs"]:
        cm = npz[f"cm_{r['tag']}"].astype(np.int64)
        out[r["tag"]] = analyse(cm, names)
        out[r["tag"]]["acc_final"] = r["acc_final"] / 100 if r["acc_final"] > 1 else r["acc_final"]

    (DATA / "error_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    for tag in ("A_BN_lr0.01", "D_noBN_lr0.05"):
        a = out[tag]
        print(f"\n=== {tag}  错 {a['wrong_total']} 张 ===")
        print("  族内:", a["inner"], " 跨族:", a["cross"])
        print("  判错流向:", [(d["name"], d["count"],
                               f"{d['share'] * 100:.1f}%") for d in a["pred_of_wrong_top"]])
        print("  最易混前 6:", [(d["true_name"], d["pred_name"], d["count"])
                                for d in a["pairs"][:6]])
        print("  涉及的类:", [(d["name"], d["wrong"], d["n_test"]) for d in a["wrong_classes"]])


if __name__ == "__main__":
    main()
