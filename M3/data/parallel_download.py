# -*- coding: utf-8 -*-
"""并行分段下载 HAM10000 v2。

要点：Kaggle 端点会 302 到签名 CDN 链接，且部分 CDN 节点会忽略
Range（返回 200 + 全文件）。因此：
  1. 先手动解析最终签名 URL，并用 bytes=0-0 测试它是否真支持 206；
  2. 每个分段请求都校验：响应必须是 206，且 Content-Range 起始位置
     与请求一致，否则丢弃重试；
  3. 分段断点续传保留，但只在 Content-Range 校验通过后追加。
"""
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = ("https://www.kaggle.com/api/v1/datasets/download/"
       "kmader/skin-cancer-mnist-ham10000")
TOTAL = 5582914511
N_SEG = 16
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ham10000.zip")
PARTS = os.path.join(HERE, "parts")
os.makedirs(PARTS, exist_ok=True)

seg_size = TOTAL // N_SEG
bounds = [(i * seg_size, (TOTAL - 1) if i == N_SEG - 1
           else (i + 1) * seg_size - 1) for i in range(N_SEG)]

UA = {"User-Agent": "Mozilla/5.0"}


def resolve_final_url():
    """跟随重定向拿到签名直链，并验证支持 Range。"""
    for _ in range(8):
        req = urllib.request.Request(URL, headers={**UA, "Range": "bytes=0-0"})
        r = urllib.request.urlopen(req, timeout=60)
        final = r.geturl()
        r.close()
        # 再对最终 URL 直发一次 Range 请求验证
        req2 = urllib.request.Request(final, headers={**UA, "Range": "bytes=0-0"})
        r2 = urllib.request.urlopen(req2, timeout=60)
        ok = (r2.status == 206 and "0-0/" in (r2.headers.get("Content-Range") or ""))
        r2.close()
        if ok:
            return final
        time.sleep(3)
    raise RuntimeError("解析到的 CDN 链接不支持 Range")


def fetch(i, final_url_holder):
    lo, hi = bounds[i]
    want = hi - lo + 1
    part = os.path.join(PARTS, f"part_{i:02d}")
    if os.path.exists(part) and os.path.getsize(part) == want:
        return f"part {i} 已完整，跳过"
    if os.path.exists(part) and os.path.getsize(part) > want:
        os.remove(part)
    for attempt in range(10):
        try:
            final_url = final_url_holder[0]
            done = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(
                final_url, headers={**UA,
                                    "Range": f"bytes={lo + done}-{hi}"})
            r = urllib.request.urlopen(req, timeout=180)
            cr = r.headers.get("Content-Range") or ""
            m = re.match(r"bytes (\d+)-(\d+)/(\d+)", cr)
            if r.status != 206 or not m or int(m.group(1)) != lo + done \
                    or int(m.group(3)) != TOTAL:
                r.close()
                raise IOError(f"part {i} 响应异常 status={r.status} "
                              f"content-range={cr!r}")
            with open(part, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            r.close()
            if os.path.getsize(part) == want:
                return f"part {i} 完成"
            raise IOError(f"part {i} 大小 {os.path.getsize(part)}/{want}")
        except Exception as e:            # noqa: BLE001
            # 签名链接可能失效，重解析
            try:
                final_url_holder[0] = resolve_final_url()
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
            if attempt == 9:
                return f"part {i} 失败: {e}"
    return f"part {i} 失败"


def main():
    t0 = time.time()
    holder = [resolve_final_url()]
    print(f"签名直链已解析，Range 验证通过", flush=True)
    with ThreadPoolExecutor(max_workers=N_SEG) as ex:
        for msg in ex.map(lambda k: fetch(k, holder), range(N_SEG)):
            print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)
    for i, (lo, hi) in enumerate(bounds):
        want = hi - lo + 1
        p = os.path.join(PARTS, f"part_{i:02d}")
        assert os.path.getsize(p) == want, f"part {i} 不完整"
    with open(OUT, "wb") as out:
        for i in range(N_SEG):
            with open(os.path.join(PARTS, f"part_{i:02d}"), "rb") as f:
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
    assert os.path.getsize(OUT) == TOTAL
    print(f"DONE {os.path.getsize(OUT)} bytes, {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
