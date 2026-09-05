# -*- coding: utf-8 -*-
"""免拼接、免落盘原始图的流式抽取。

从 16 个分段直接虚拟拼接读取 zip，边读边把需要的图预处理成 npz：
  - ham_gray96.npz：mel 全部 1113 张 + nv 抽样 1000 张，96×96 灰度 uint8
  - showcase/：7 类各 3 张 224×224 RGB jpg（fig1 展示用）
  - HAM10000_metadata.csv + extracted_manifest.json
磁盘足迹约 25MB。
"""
import csv
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "parts")
SHOW = os.path.join(HERE, "showcase")
os.makedirs(SHOW, exist_ok=True)
N_SEG = 16


class PartsReader(io.RawIOBase):
    """把 N 个分段文件呈现为一个只读连续字节流。"""

    def __init__(self, paths):
        self.paths = paths
        self.sizes = [os.path.getsize(p) for p in paths]
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.total = self.offsets[-1]
        self.pos = 0
        self._fh = None
        self._fh_idx = -1

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total + offset
        return self.pos

    def tell(self):
        return self.pos

    def _open_part(self, idx):
        if self._fh_idx != idx:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self.paths[idx], "rb")
            self._fh_idx = idx
        return self._fh

    def _idx_of(self, pos):
        lo, hi = 0, len(self.offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.offsets[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.total:
            idx = self._idx_of(self.pos)
            fh = self._open_part(idx)
            fh.seek(self.pos - self.offsets[idx])
            avail = self.offsets[idx + 1] - self.pos
            chunk = fh.read(min(n, avail))
            if not chunk:
                break
            out.extend(chunk)
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self):
        if self._fh is not None:
            self._fh.close()
        super().close()


def main():
    paths = [os.path.join(PARTS, f"part_{i:02d}") for i in range(N_SEG)]
    for p in paths:
        assert os.path.exists(p), p
    reader = PartsReader(paths)
    print(f"虚拟流总长 {reader.total/1e9:.2f} GB")
    zf = zipfile.ZipFile(reader)
    names = zf.namelist()
    meta_name = [n for n in names if n.endswith("HAM10000_metadata.csv")][0]
    img_names = [n for n in names
                 if n.startswith(("HAM10000_images_part_1/",
                                  "HAM10000_images_part_2/"))]
    name_of = {os.path.splitext(os.path.basename(n))[0]: n
               for n in img_names}

    rows = list(csv.DictReader(
        zf.read(meta_name).decode("utf-8").splitlines()))
    print(f"metadata: {len(rows)} 条")
    by_dx = {}
    for r in rows:
        by_dx.setdefault(r["dx"], []).append(r["image_id"])

    rng42 = np.random.default_rng(42)
    mel_ids = list(by_dx["mel"])
    nv_ids = [by_dx["nv"][i] for i in
              rng42.choice(len(by_dx["nv"]), size=1000, replace=False)]
    train_ids = mel_ids + nv_ids
    train_y = np.array([1] * len(mel_ids) + [0] * len(nv_ids),
                       dtype=np.int64)

    rng3 = np.random.default_rng(3)
    showcase = {}
    for dx in by_dx:
        picks = [by_dx[dx][int(rng3.integers(len(by_dx[dx])))]
                 for _ in range(3)]
        showcase[dx] = picks

    # 预处理：训练图 96×96 灰度；展示图 224×224 RGB
    G = np.empty((len(train_ids), 96, 96), dtype=np.uint8)
    for k, iid in enumerate(train_ids):
        img = Image.open(io.BytesIO(zf.read(name_of[iid])))
        G[k] = np.asarray(img.convert("L").resize(
            (96, 96), Image.BILINEAR), dtype=np.uint8)
        if (k + 1) % 500 == 0:
            print(f"  训练图 {k+1}/{len(train_ids)}", flush=True)
    np.savez_compressed(os.path.join(HERE, "ham_gray96.npz"),
                        ids=np.array(train_ids), y=train_y, X=G)
    for dx, iids in showcase.items():
        for iid in iids:
            img = Image.open(io.BytesIO(zf.read(name_of[iid])))
            img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
            img.save(os.path.join(SHOW, f"{dx}_{iid}.jpg"), quality=90)

    with open(os.path.join(HERE, "HAM10000_metadata.csv"), "wb") as f:
        f.write(zf.read(meta_name))
    meta = {
        "mel_ids": mel_ids, "nv_sampled": sorted(nv_ids),
        "showcase": showcase, "n_train": len(train_ids),
        "counts": {dx: len(v) for dx, v in by_dx.items()},
    }
    with open(os.path.join(HERE, "extracted_manifest.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"完成：ham_gray96.npz({len(train_ids)} 张 96×96 灰度) "
          f"+ showcase {sum(len(v) for v in showcase.values())} 张")


if __name__ == "__main__":
    main()
