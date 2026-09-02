# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, itertools, os
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

for f in ["STHeiti Medium","Heiti SC","PingFang SC","Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"]=[f]; break
    except Exception: continue
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False,
                     "font.size":11,"axes.titlesize":13,"axes.titleweight":"bold","axes.labelsize":11})
OKA={"orange":"#E69F00","blue":"#56B4E9","green":"#009E73","yellow":"#F0E442",
     "dblue":"#0072B2","red":"#D55E00","pink":"#CC79A7","grey":"#999999"}
os.makedirs("figures",exist_ok=True)

cols=["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship",
      "race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
df=pd.read_csv("data/adult.data",names=cols,skipinitialspace=True,na_values="?")
hi=(df.income==">50K").astype(int).values; edn=df.education_num.values.astype(float)
hrs=df.hours_per_week.values.astype(float); N=len(df)
P=lambda x,y: float(np.corrcoef(x,y)[0,1])

# ------------------------------------------------------------------ fig1
combos=list(itertools.combinations(range(1,19),5))
W=np.array([sum(c) for c in combos]); C=len(W); w_obs=80; w_mirror=15
mu=5*19/2; sd=np.sqrt(5*13*19/12)
fig,ax=plt.subplots(figsize=(9,4.8))
bins=np.arange(W.min()-0.5,W.max()+1.5,1)
cnt,edges,patches=ax.hist(W,bins=bins,color=OKA["blue"],alpha=.85,edgecolor="white",linewidth=.3,
                          label=f"{C} 种选法的精确零分布")
xs=np.linspace(W.min(),W.max(),600)
ax.plot(xs,stats.norm.pdf(xs,mu,sd)*C*1.0,color=OKA["red"],lw=2.2,
        label=f"正态近似 N({mu}, {sd:.2f}²)")
ax.axvline(w_obs,color=OKA["orange"],lw=2.4,ls="-",zorder=5)
ax.axvline(w_mirror,color=OKA["grey"],lw=1.8,ls="--",zorder=5)
ax.axvline(mu,color="#333333",lw=1.2,ls=":",zorder=4)
ax.annotate(f"观测值 W = {w_obs}\n(后 5 名全部入选)\n精确 p = 1/{C} = 1.17e-4",
            xy=(w_obs,ax.get_ylim()[1]*0.72),xytext=(w_obs-22,ax.get_ylim()[1]*0.80),
            fontsize=10,color=OKA["orange"],fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=OKA["orange"],lw=1.6))
ax.annotate(f"对称的另一端 W = {w_mirror}\n(前 5 名全部入选)",xy=(w_mirror,30),
            xytext=(w_mirror+3,ax.get_ylim()[1]*0.52),fontsize=9.5,color=OKA["grey"],
            arrowprops=dict(arrowstyle="->",color=OKA["grey"],lw=1.2))
ax.annotate(f"期望 E[W] = {mu}",xy=(mu,ax.get_ylim()[1]*0.93),xytext=(mu-9,ax.get_ylim()[1]*0.97),
            fontsize=9.5,color="#333333")
ax.annotate("正态近似在这里\n明显不够尖\n（尾部概率被低估 5.8 倍）",
            xy=(w_obs-4,stats.norm.pdf(w_obs,mu,sd)*C*1.0+4),
            xytext=(w_obs-30,ax.get_ylim()[1]*0.32),fontsize=9.5,color=OKA["red"],
            arrowprops=dict(arrowstyle="->",color=OKA["red"],lw=1.2))
ax.set_xlabel("入选 5 人的笔试名次之和 W"); ax.set_ylabel("出现该 W 的选法数量")
ax.set_title("图1  如果面试与笔试无关：18 选 5 的 8568 种可能里，观测值有多靠边")
ax.legend(loc="upper left",frameon=False,fontsize=9.5)
ax.set_xlim(W.min()-2,W.max()+2)
plt.tight_layout(); plt.savefig("figures/fig1_null_dist.png",bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ fig2
fig,ax=plt.subplots(figsize=(9,4.8))
xs=np.linspace(-5,5,1200)
for ddf,c,lab in [(2,OKA["red"],"t (df=2)"),(5,OKA["orange"],"t (df=5)"),
                  (37.71,OKA["green"],"t (df=37.7)"),(None,"#333333","标准正态 Z")]:
    y=stats.norm.pdf(xs) if ddf is None else stats.t.pdf(xs,ddf)
    ax.plot(xs,y,color=c,lw=2.2 if ddf is None else 1.9,label=lab,
            ls="-" if ddf is None else "--")
crit_n=1.96; crit_t38=2.0249; crit_t5=2.5706; crit_t2=4.3027
ax.axvline(crit_n,color="#333333",lw=1,ls=":",alpha=.8)
ax.axvline(crit_t38,color=OKA["green"],lw=1,ls=":",alpha=.8)
ax.axvline(crit_t5,color=OKA["orange"],lw=1,ls=":",alpha=.8)
ax.annotate(f"正态 ±1.96",xy=(crit_n,.40),xytext=(crit_n+.15,.44),fontsize=9,color="#333333")
ax.annotate(f"t(37.7) ±2.025",xy=(crit_t38,.30),xytext=(crit_t38+.15,.34),fontsize=9,color=OKA["green"])
ax.annotate(f"t(5) ±2.571",xy=(crit_t5,.18),xytext=(crit_t5+.15,.22),fontsize=9,color=OKA["orange"])
ax.annotate("自由度越小，尾巴越厚\n临界值越往外推\n（同样的差值更难显著）",
            xy=(-4.0,.055),fontsize=9.5,color=OKA["red"],
            bbox=dict(boxstyle="round,pad=0.45",fc="white",ec=OKA["red"],alpha=.9))
ax.fill_between(xs[xs>=crit_n],0,stats.norm.pdf(xs[xs>=crit_n]),color="#333333",alpha=.10)
ax.fill_between(xs[xs>=crit_t5],0,stats.t.pdf(xs[xs>=crit_t5],5),color=OKA["orange"],alpha=.14)
ax.set_xlabel("检验统计量"); ax.set_ylabel("概率密度")
ax.set_title("图2  为什么样本小的时候不能拿 1.96 当门槛：t 分布比正态分布胖")
ax.set_xlim(-5,5); ax.set_ylim(0,.46); ax.legend(loc="upper right",frameon=False,fontsize=9.5)
plt.tight_layout(); plt.savefig("figures/fig2_t_vs_z.png",bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ fig3
labels=["初中及以下\n(1-9年)","高中/部分大学\n(10-11年)","学士\n(13年)","硕士及以上\n(14-16年)"]
O=np.array([[12835,1919],[5904,1387],[4957,2847],[1024,1688]],dtype=float)
E=np.outer(O.sum(1),O.sum(0))/O.sum()
rate=O[:,1]/O.sum(1); rate_all=O[:,1].sum()/O.sum()
contrib=(O-E)**2/E
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.9),gridspec_kw={"width_ratios":[1.25,1]})
xpos=np.arange(4)
b=ax1.bar(xpos,rate*100,color=[OKA["blue"],OKA["blue"],OKA["orange"],OKA["red"]],
          width=.62,edgecolor="white")
ax1.axhline(rate_all*100,color=OKA["red"],ls="--",lw=1.8)
ax1.annotate(f"独立性假设下的期望值 {rate_all*100:.2f}%",xy=(2.6,rate_all*100+1.6),
             fontsize=9.5,color=OKA["red"])
for i,(r_,n_) in enumerate(zip(rate,O.sum(1))):
    ax1.text(i,r_*100+1.1,f"{r_*100:.2f}%",ha="center",fontsize=10.5,fontweight="bold")
    ax1.text(i,2.2,f"n={int(n_)}",ha="center",fontsize=8.5,color="white")
ax1.set_xticks(xpos); ax1.set_xticklabels(labels,fontsize=9.5)
ax1.set_ylabel("高收入(>50K)占比 %"); ax1.set_ylim(0,72)
ax1.set_title(f"观测：学历越高，高收入占比越高",fontsize=12)
im=ax2.imshow(contrib,cmap="YlOrRd",aspect="auto")
ax2.set_xticks([0,1]); ax2.set_xticklabels(["≤50K",">50K"],fontsize=10)
ax2.set_yticks(range(4)); ax2.set_yticklabels(["初中及以下","高中/部分大学","学士","硕士及以上"],fontsize=9.5)
for i in range(4):
    for j in range(2):
        v=contrib[i,j]
        ax2.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=11,fontweight="bold",
                 color="white" if v>contrib.max()*0.45 else "#333")
ax2.set_title(f"每个格子对 χ² 的贡献\n(总和 χ² = {contrib.sum():.0f}, df = 3)",fontsize=12)
plt.colorbar(im,ax=ax2,shrink=.78,label=r"$(O-E)^2/E$")
plt.tight_layout(); plt.savefig("figures/fig3_chi2.png",bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ fig4
rng=np.random.default_rng(1)
sub=rng.choice(N,200,replace=False); xe,ye=edn[sub].copy(),hrs[sub].copy()
p0,s0=P(xe,ye),float(stats.spearmanr(xe,ye).statistic)
rec=[{"k":0,"p":p0,"s":s0}]
for k in [1,3]:
    xo=np.r_[xe,[1.]*k]; yo=np.r_[ye,[99.]*k]
    rec.append({"k":k,"p":P(xo,yo),"s":float(stats.spearmanr(xo,yo).statistic)})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.9),gridspec_kw={"width_ratios":[1.3,1]})
ax1.scatter(xe,ye,s=16,alpha=.28,color=OKA["blue"],label="200 人原始样本")
ax1.scatter([1,1,1],[99,99,99],s=90,color=OKA["red"],marker="X",zorder=6,
            label="人为塞入的 3 个极端点\n(1 年教育, 每周 99 小时)")
b1,a1=np.polyfit(xe,ye,1)
xs=np.linspace(0,17,50); ax1.plot(xs,b1*xs+a1,color=OKA["dblue"],lw=2,label=f"Pearson 拟合线 (r={p0:.3f})")
xo=np.r_[xe,[1.]*3]; yo=np.r_[ye,[99.]*3]
b2,a2=np.polyfit(xo,yo,1); ax1.plot(xs,b2*xs+a2,color=OKA["red"],lw=2,ls="--",
                                    label=f"加 3 个点后 (r={rec[2]['p']:.3f})")
ax1.set_xlabel("受教育年限"); ax1.set_ylabel("每周工作小时")
ax1.set_title("3 个点把 Pearson 的符号都拽反了",fontsize=12)
ax1.legend(loc="lower right",frameon=False,fontsize=8.8)
ks=[r_["k"] for r_ in rec]; ps=[r_["p"] for r_ in rec]; ss=[r_["s"] for r_ in rec]
w_=.38; xpos=np.arange(3)
ax2.bar(xpos-w_/2,ps,w_,color=OKA["dblue"],label="Pearson r",edgecolor="white")
ax2.bar(xpos+w_/2,ss,w_,color=OKA["green"],label="Spearman ρ",edgecolor="white")
ax2.axhline(0,color="#333333",lw=1)
for i,(v1,v2) in enumerate(zip(ps,ss)):
    ax2.text(i-w_/2,v1+(0.018 if v1>=0 else -0.032),f"{v1:+.3f}",ha="center",fontsize=9.5,fontweight="bold",color=OKA["dblue"])
    ax2.text(i+w_/2,v2+(0.018 if v2>=0 else -0.032),f"{v2:+.3f}",ha="center",fontsize=9.5,fontweight="bold",color=OKA["green"])
ax2.set_xticks(xpos); ax2.set_xticklabels(["原始 200 人","加 1 个极端点","加 3 个极端点"],fontsize=9.5)
ax2.set_ylabel("相关系数"); ax2.set_ylim(-0.14,0.22)
ax2.set_title("同一个样本，两个系数的抗揍程度",fontsize=12)
ax2.legend(loc="lower left",frameon=False,fontsize=9.5)
plt.tight_layout(); plt.savefig("figures/fig4_corr_outlier.png",bbox_inches="tight"); plt.close()
print("4 张图已生成:")
for f_ in sorted(os.listdir("figures")): print("  figures/"+f_)
