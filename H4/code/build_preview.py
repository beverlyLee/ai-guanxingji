#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 H4 正文渲染成自包含 HTML（图片 base64 内嵌），用于发预览链接。"""
import os, re, base64
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "H4_线性代数_推荐系统_信息茧房_SVD_2026-08-25.md")
ASSETS = os.path.join(HERE, "assets")
OUT_DIR = os.path.join(HERE, "发布预览")
OUT = os.path.join(OUT_DIR, "index.html")

os.makedirs(OUT_DIR, exist_ok=True)

with open(SRC, encoding="utf-8") as f:
    md = f.read()

# 把本地图片引用替换成 base64 data URI
def embed(match):
    alt, path = match.group(1), match.group(2)
    fp = os.path.join(HERE, path) if not os.path.isabs(path) else path
    if not os.path.exists(fp):
        fp = os.path.join(ASSETS, os.path.basename(path))
    if not os.path.exists(fp):
        return match.group(0)
    ext = fp.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(ext, "image/png")
    b64 = base64.b64encode(open(fp, "rb").read()).decode()
    return f'<img alt="{alt}" src="data:image/{mime};base64,{b64}" />'

md = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', embed, md)

html_body = markdown.markdown(
    md,
    extensions=["tables", "fenced_code", "toc", "nl2br"],
    extension_configs={"toc": {"permalink": False}},
)

CSS = """
:root{--ink:#1a1a1a;--sub:#666;--bg:#fbfaf7;--card:#fff;--accent:#D55E00;--blue:#0072B2;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  line-height:1.85;font-size:17px;}
.wrap{max-width:760px;margin:0 auto;padding:48px 22px 96px;}
h1{font-size:30px;line-height:1.35;margin:0 0 8px;letter-spacing:.3px;}
h2{font-size:23px;margin:42px 0 14px;padding-left:12px;border-left:5px solid var(--accent);}
h3{font-size:19px;margin:30px 0 10px;color:var(--accent);}
p{margin:16px 0;}
img{max-width:100%;height:auto;border-radius:10px;margin:22px 0;
  box-shadow:0 6px 22px rgba(0,0,0,.10);background:#fff;}
blockquote{margin:18px 0;padding:12px 18px;background:#fff;border-left:4px solid var(--blue);
  border-radius:0 8px 8px 0;color:var(--sub);}
code{background:#f0eee9;padding:2px 6px;border-radius:5px;font-size:14px;}
ul,ol{padding-left:24px;} li{margin:8px 0;}
hr{border:none;border-top:1px solid #e6e2da;margin:40px 0;}
.meta{color:var(--sub);font-size:14px;margin-bottom:6px;}
.tag{display:inline-block;background:#fff;border:1px solid #e6e2da;border-radius:999px;
  padding:3px 12px;font-size:13px;color:var(--sub);margin-right:8px;}
.foot{margin-top:60px;padding-top:22px;border-top:1px solid #e6e2da;color:var(--sub);font-size:13px;}
"""

title_match = re.search(r'^#\s+(.+)$', md, re.M)
title = title_match.group(1) if title_match else "AI观星记 · H4 信息茧房 / SVD"

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<div class="meta">AI观星记 · 第 4 篇 · 线性代数 / 推荐系统 / 信息茧房（SVD 奇异值分解）</div>
{html_body}
<div class="foot">本页由 H4 正文自动渲染，含 4 张由真实 MovieLens 100K 数据生成的配图（fig1–fig4）。
数据与复现脚本见专栏「AI观星记」。</div>
</div>
</body>
</html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

print("已生成:", OUT, " 大小:", os.path.getsize(OUT), "字节")
