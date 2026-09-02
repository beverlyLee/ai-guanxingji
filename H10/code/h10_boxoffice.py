# -*- coding: utf-8 -*-
"""
H10《牛来》票房涨 1000 倍是真的吗？——2026 暑期档 124 亿里的数学
所有数字均来自国家电影局 / 猫眼研究院 / 灯塔专业版公开通报（新浪、腾讯、解放日报报道）。
脚本目的：复现文章中的每个计算，并生成 4 张配图（Okabe-Ito 色盲安全配色 + 中文 Heiti SC）。
"""
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Heiti SC', 'PingFang SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300

import matplotlib.pyplot as plt
from matplotlib import patheffects

# ---- Okabe-Ito 色盲安全配色 ----
OI = {
    'black':      '#000000',
    'orange':     '#E69F00',
    'skyblue':    '#56B4E9',
    'bluishgrn':  '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'redpurple':  '#CC79A7',
}

FIGDIR = 'figures'

# ============================================================
# 0. 原始公开数据（2026 暑期档，6.1–8.31）
# ============================================================
total_box   = 124.98      # 亿，国家电影局
growth_box  = 0.0445      # 同比 +4.45%
audience    = 3.4         # 亿人次
growth_aud  = 0.0586      # 同比 +5.86%
shows       = 3851        # 万场（历史新高）
avg_price   = 36.74        # 元（2022 以来最低）
price_drop  = 0.50         # 比去年降约 5 角
top3        = [23.15, 19.51, 18.46]   # 亿：功夫女足/欢迎来龙餐馆/八仙!
top5        = top3 + [15.39, 6.03]     # +蜘蛛侠/奥德赛
occupancy   = 7.21         # 上座率 %
occ_last    = 6.94
per_show    = 8.8          # 场均人次
imax_box    = 7.45         # IMAX 总票房 亿
imax_growth = 0.40         # IMAX 同比 +40%
od_imax     = 3.28         # 奥德赛 IMAX 亿
od_total    = 6.03         # 奥德赛总票房 亿
male_share  = 43.8         # 男性观众占比 %
male_last   = 40.2

# 《牛来》公开锚点（灯塔/猫眼/媒体报道）
nl_day1     = 3420         # 首日票房 元（另有媒体报 342 元，口径不一）
nl_cum_d9   = 7169         # 前 9 天累计 元
nl_peak_day = 260000       # 8.15 单日约 元（报道 18–26 万，取中值）
nl_cum_aug  = 12_600_000   # 截至 8.31 累计 元（灯塔口径；腾讯报近 6000 万，口径不一）
nl_predict  = 90_000_000   # 猫眼预测终值 元

print('=' * 60)
print('Q1 《牛来》倍数核验')
print('=' * 60)
cum_mult   = nl_cum_aug / nl_cum_d9
single_mult = nl_peak_day / nl_day1
pred_mult  = nl_predict / nl_day1
print(f'  累计倍数   = {nl_cum_aug:,} / {nl_cum_d9:,} = {cum_mult:,.0f} 倍')
print(f'  单日倍数   = {nl_peak_day:,} / {nl_day1:,} = {single_mult:,.0f} 倍')
print(f'  预测/首日  = {nl_predict:,} / {nl_day1:,} = {pred_mult:,.0f} 倍')

# 指数拟合 S(t)=a*exp(r*t)，用首日与 8.31 累计两端点（8.5->8.31 共 27 天）
t1, t2 = 1, 27
ratio = nl_cum_aug / nl_day1
r = np.log(ratio) / (t2 - t1)
print(f'  指数拟合日复合增速 r = ln({ratio:,.0f})/26 = {r*100:.1f} %/天')

# ============================================================
# Q2 以价换量：需求价格弹性
# ============================================================
print('\n' + '=' * 60)
print('Q2 需求价格弹性')
print('=' * 60)
price_last   = avg_price + price_drop
pct_price    = (avg_price - price_last) / price_last
aud_last     = audience / (1 + growth_aud)
pct_aud      = (audience - aud_last) / aud_last
elasticity   = pct_aud / pct_price
rev_check    = pct_price + pct_aud   # 一阶近似 %ΔRev
print(f'  去年均价   = {price_last:.2f} 元；今年 {avg_price} 元 -> 价变动 {pct_price*100:.2f}%')
print(f'  去年人次   = {aud_last:.3f} 亿；今年 {audience} 亿 -> 量变动 {pct_aud*100:.2f}%')
print(f'  弹性 ε     = {pct_aud*100:.2f}% / ({pct_price*100:.2f}%) = {elasticity:.2f}  (|ε|>1 富有弹性)')
print(f'  一阶校验 %ΔRev≈%ΔP+%ΔQ = {rev_check*100:.2f}%  (公布 +4.45%)')

# ============================================================
# Q3 核验 + 集中度
# ============================================================
print('\n' + '=' * 60)
print('Q3 人次/场次核验 + 市场集中度')
print('=' * 60)
calc_per_show = audience * 1e8 / (shows * 1e4)
print(f'  场均 = {audience}e8 / ({shows}e4) = {calc_per_show:.2f} 人/场  (公布 {per_show})')
cr3 = sum(top3) / total_box * 100
cr5 = sum(top5) / total_box * 100
print(f'  CR3 = {sum(top3):.2f}/{total_box} = {cr3:.1f}%   CR5 = {sum(top5):.2f}/{total_box} = {cr5:.1f}%')

# HHI（示意）：Top5 已知份额，剩余 42.44 亿假设均摊给 40 部影片
shares_top5 = np.array(top5) / total_box
rest = (total_box - sum(top5)) / total_box
n_rest = 40
hhi = np.sum(shares_top5**2) + n_rest * (rest / n_rest)**2
print(f'  HHI(示意, 剩余均摊{n_rest}部) = {hhi*10000:,.0f}  (中等集中；>2500 为高集中)')

# ============================================================
# Q4 IMAX 结构
# ============================================================
print('\n' + '=' * 60)
print('Q4 IMAX 结构性涨价')
print('=' * 60)
od_imax_share = od_imax / od_total
imax_share_total = imax_box / total_box
imax_last = imax_box / (1 + imax_growth)
delta_imax = imax_box - imax_last
delta_total = total_box - total_box / (1 + growth_box)
print(f'  奥德赛 IMAX 占比 = {od_imax}/{od_total} = {od_imax_share*100:.1f}%')
print(f'  IMAX 总占大盘   = {imax_box}/{total_box} = {imax_share_total*100:.1f}%')
print(f'  IMAX 增量 {delta_imax:.2f} 亿 占 大盘增量 {delta_total:.2f} 亿 = {delta_imax/delta_total*100:.1f}%')

# ============================================================
# 配图
# ============================================================
def despine(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

# ---- fig1：《牛来》指数增长（累计票房，对数纵轴）----
fig, ax = plt.subplots(figsize=(7, 4.2))
days = np.arange(1, 28)
S = nl_day1 * np.exp(r * (days - 1))
ax.plot(days, S, color=OI['vermillion'], lw=2.2, label=f'指数拟合 (日增 {r*100:.0f}%)')
ax.scatter([1, 9, 27], [nl_day1, nl_cum_d9, nl_cum_aug],
           color=OI['blue'], zorder=5, s=45)
ax.scatter([1], [nl_day1], color=OI['blue'], s=45)
for x, y, txt in [(1, nl_day1, '首日 3420 元'),
                 (9, nl_cum_d9, '前9天 7169 元'),
                 (27, nl_cum_aug, '8.31 累计 1260 万')]:
    ax.annotate(txt, (x, y), textcoords='offset points', xytext=(8, 6),
                fontsize=9, color=OI['black'])
ax.set_yscale('log')
ax.set_xlabel('上映天数（自 8.5 起）')
ax.set_ylabel('累计票房（元，对数轴）')
ax.set_title('《牛来》：从 3420 元到 1260 万，约 1760 倍', fontsize=12)
ax.axhline(nl_peak_day, color=OI['bluishgrn'], ls='--', lw=1, alpha=.7)
ax.annotate(f'单日峰值 ≈ 26 万\n(约为首日的 {single_mult:.0f} 倍)',
            (15, nl_peak_day), textcoords='offset points', xytext=(10, -28),
            fontsize=9, color=OI['bluishgrn'])
despine(ax)
ax.legend(frameon=False, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig1_niulai_growth.png', bbox_inches='tight')
plt.close()

# ---- fig2：以价换量（去年 vs 今年）----
fig, ax = plt.subplots(figsize=(7, 4.2))
metrics = ['平均票价(元)', '观影人次(亿)', '总票房(亿)']
last = [price_last, aud_last, total_box/(1+growth_box)]
now  = [avg_price, audience, total_box]
x = np.arange(len(metrics)); w = 0.38
b1 = ax.bar(x - w/2, last, w, color=OI['skyblue'], label='去年')
b2 = ax.bar(x + w/2, now,  w, color=OI['vermillion'], label='今年')
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f'{b.get_height():.2f}', (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title('以价换量：票价降 1.3%，人次涨 5.9%，收入反升', fontsize=12)
ax.annotate(f'需求弹性 ε ≈ {elasticity:.1f}（富有弹性）', (2, total_box*0.5),
            fontsize=9.5, color=OI['blue'])
despine(ax)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig2_price_elasticity.png', bbox_inches='tight')
plt.close()

# ---- fig3：头部集中度 ----
fig, ax = plt.subplots(figsize=(7, 4.2))
names = ['功夫女足', '欢迎来龙\n餐馆', '八仙!', '蜘蛛侠', '奥德赛']
vals = top5
colors = [OI['orange'], OI['skyblue'], OI['bluishgrn'], OI['blue'], OI['redpurple']]
bars = ax.bar(names, vals, color=colors)
for b, v in zip(bars, vals):
    ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), textcoords='offset points',
                xytext=(0, 3), ha='center', fontsize=9)
ax.set_ylabel('票房（亿元）')
ax.set_title('头部集中：Top3 占 49%，Top5 占 66%', fontsize=12)
ax.annotate(f'CR3={cr3:.0f}%  CR5={cr5:.0f}%\nHHI≈{hhi*10000:,.0f}（中等集中）',
            (3.6, 22.5), fontsize=9, color=OI['black'], ha='right')
despine(ax)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig3_concentration.png', bbox_inches='tight')
plt.close()

# ---- fig4：IMAX 结构性涨价 ----
fig, ax = plt.subplots(figsize=(7, 4.2))
labels = ['奥德赛片内\nIMAX 占比', 'IMAX 总票房\n占大盘']
vals = [od_imax_share*100, imax_share_total*100]
bars = ax.bar(labels, vals, color=[OI['vermillion'], OI['blue']], width=0.5)
for b, v in zip(bars, vals):
    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v), textcoords='offset points',
                xytext=(0, 4), ha='center', fontsize=10)
ax.set_ylabel('占比（%）')
ax.set_title('高端格式对冲均价下跌：IMAX 同比 +40%', fontsize=12)
ax.annotate(f'IMAX 增量占大盘增量约 {delta_imax/delta_total*100:.0f}%',
            (0.5, 30), fontsize=9, color=OI['black'])
despine(ax)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig4_imax_share.png', bbox_inches='tight')
plt.close()

print('\nFigures saved to', FIGDIR)
