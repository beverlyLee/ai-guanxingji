# -*- coding: utf-8 -*-
"""
AI观星记 · M4 · 词向量与 RNN 落地代码
手写 Skip-gram(负采样) / CBOW / 全 softmax / RNN(BPTT) / 残差式RNN(直通路对照)，含数值梯度校验。
数据：online_shopping_10_cats（中文电商评论 62,774 条，10 品类，正 31,728 / 负 31,046）
产出：stats.json + figures/fig1..fig8（全部从数据重生，Okabe-Ito 配色，STHeiti 中文）
"""
import os, sys, json, math, re, time, csv
from collections import Counter, defaultdict
import numpy as np

ROOT = "/Users/liboyang/WorkBuddy/2026-08-13-17-53-26/AI观星记_M4_词向量与RNN"
DATA = os.path.join(ROOT, "data", "online_shopping_10_cats.csv")
FIG  = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SEED = 20260906
np.random.seed(SEED)

# ---------- 可调规模（受算力约束，全部用真实数据的子样本/分层抽样，正文如实说明）----------
MAX_VOCAB      = 6000
MIN_COUNT      = 5
N_TRAIN_W2V    = 6000      # 训练词向量用的评论数（按品类分层抽样，覆盖全部 10 类）
W2V_DIM        = 40
W2V_WINDOW     = 4
W2V_K          = 5
W2V_EPOCHS     = 1
W2V_LR         = 0.03
N_SENT         = 6000
RNN_HID        = 40
RNN_EPOCHS     = 15
RNN_LR         = 0.02
CATEGORIES     = ["书籍","平板","手机","水果","洗发水","热水器","蒙牛","衣服","计算机","酒店"]

# ---------- Okabe-Ito 色盲安全配色 ----------
C = ["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2",
     "#D55E00","#CC79A7","#999999","#882255"]
FONTPATH = "/System/Library/Fonts/STHeiti Light.ttc"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
if os.path.exists(FONTPATH):
    font_manager.fontManager.addfont(FONTPATH)
    plt.rcParams["font.family"] = "Heiti SC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

def style_ax(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name)); plt.close()
    print("  saved", name)

def sigmoid(x):
    x=np.asarray(x,float); x=np.clip(x,-30,30); return 1.0/(1.0+np.exp(-x))
def cos(a, b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); return float(a@b)

# =====================================================================
# 载入 + 分词 + 建表
# =====================================================================
print("[EXP-1] 载入语料与分词 ...")
import jieba
jieba.setLogLevel(20)

rows = []
with open(DATA, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        rows.append((row["cat"], int(row["label"]), row["review"]))
print("  总评论数 =", len(rows))

def tok(s):
    out=[]
    for t in jieba.lcut(s):
        t=t.strip().lower()
        if not t: continue
        if re.fullmatch(r"[\s\W\d]+", t): continue
        if not re.search(r"[\u4e00-\u9fff]", t): continue
        out.append(t)
    return out

all_tokens=[]; cat_tokens=defaultdict(list); per_cat_reviews=defaultdict(list)
for cat, lab, rev in rows:
    per_cat_reviews[cat].append((lab, rev))
    ts=tok(rev); all_tokens.append(ts); cat_tokens[cat].append(ts)
flat=[t for ts in all_tokens for t in ts]
freq=Counter(flat)
print("  总 token 数 =", len(flat), " 原始词种 =", len(freq))

vocab=[w for w,c in freq.most_common(MAX_VOCAB) if c>=MIN_COUNT][:MAX_VOCAB]
word2id={w:i for i,w in enumerate(vocab)}; id2word={i:w for i,w in enumerate(vocab)}
V=len(vocab); print("  词表大小 V =", V)

freq_arr=np.array([freq[w] for w in vocab], dtype=np.float64)
rank=np.arange(1,V+1)
zipf_corr=float(np.corrcoef(np.log(rank), np.log(freq_arr+1))[0,1])
noise=freq_arr**0.75; noise/=noise.sum()
NEG_TABLE=np.random.choice(V, size=10_000_000, p=noise)
NEG_TABLE_RV=None
def sample_neg(K):
    tab=NEG_TABLE_RV if NEG_TABLE_RV is not None else NEG_TABLE
    return tab[np.random.randint(0,len(tab),size=K)]

EXP1={"total_reviews":len(rows),"total_tokens":len(flat),"vocab_size":V,
      "zipf_loglog_corr":round(zipf_corr,4),
      "top20_words":[[w,int(freq[w])] for w in vocab[:20]],
      "min_count":MIN_COUNT,"max_vocab":MAX_VOCAB}

def to_ids(ts): return [word2id[t] for t in ts if t in word2id]
all_ids=[to_ids(ts) for ts in all_tokens]
cat_ids={cat:[to_ids(ts) for ts in cat_tokens[cat]] for cat in CATEGORIES}

per_cat_n=max(1,N_TRAIN_W2V//len(CATEGORIES))
w2v_corpus=[]
for cat in CATEGORIES:
    revs=per_cat_reviews[cat][:]; np.random.shuffle(revs)
    for lab,rev in revs[:per_cat_n]:
        ids=to_ids(tok(rev))
        if len(ids)>=2: w2v_corpus.append(ids)
print("  词向量训练语料句子数 =", len(w2v_corpus))

# =====================================================================
# EXP-3/4 ：手写 Skip-gram(负采样) 与 CBOW
# =====================================================================
def init_params(V,d):
    s=0.5/np.sqrt(d)
    return np.random.uniform(-s,s,(V,d)), np.random.uniform(-s,s,(V,d))

def train_skipgram(corpus,V,d,window,K,epochs,lr):
    W_in,W_out=init_params(V,d); t0=time.time()
    for ep in range(epochs):
        for ids in corpus:
            L=len(ids)
            for i in range(L):
                c=ids[i]; v=W_in[c]
                lo,hi=max(0,i-window),min(L,i+window+1)
                for j in range(lo,hi):
                    if j==i: continue
                    o=ids[j]; z_o=W_out[o]@v; s_o=sigmoid(z_o)
                    negs=sample_neg(K); Z_n=W_out[negs]@v; s_n=sigmoid(Z_n)
                    gv=(s_o-1.0)*W_out[o]+(s_n@W_out[negs])
                    W_in[c]-=lr*gv
                    W_out[o]-=lr*(s_o-1.0)*v
                    W_out[negs]-=lr*(s_n[:,None]*v[None,:])
    return W_in,W_out,time.time()-t0

def train_cbow(corpus,V,d,window,K,epochs,lr):
    W_in,W_out=init_params(V,d); t0=time.time()
    for ep in range(epochs):
        for ids in corpus:
            L=len(ids)
            for i in range(L):
                c=ids[i]
                lo,hi=max(0,i-window),min(L,i+window+1)
                ctx=[ids[j] for j in range(lo,hi) if j!=i]
                if not ctx: continue
                H=np.mean(W_in[ctx],axis=0)
                z_c=W_out[c]@H; s_c=sigmoid(z_c)
                negs=sample_neg(K); Z_n=W_out[negs]@H; s_n=sigmoid(Z_n)
                gH=(s_c-1.0)*W_out[c]+(s_n@W_out[negs])
                W_in[ctx]-=lr*gH/len(ctx)
                W_out[c]-=lr*(s_c-1.0)*H
                W_out[negs]-=lr*(s_n[:,None]*H[None,:])
    return W_in,W_out,time.time()-t0

print("[EXP-3/4] 训练 Skip-gram 与 CBOW ...")
W_in_sg,W_out_sg,t_sg=train_skipgram(w2v_corpus,V,W2V_DIM,W2V_WINDOW,W2V_K,W2V_EPOCHS,W2V_LR)
W_in_cb,W_out_cb,t_cb=train_cbow(w2v_corpus,V,W2V_DIM,W2V_WINDOW,W2V_K,W2V_EPOCHS,W2V_LR)
print("  Skip-gram 耗时 %.1fs, CBOW 耗时 %.1fs"%(t_sg,t_cb))

W=W_in_sg.copy(); Wn=W/np.linalg.norm(W,axis=1,keepdims=True)
def nearest(w,k=8):
    if w not in word2id: return []
    q=Wn[word2id[w]]; sims=Wn@q; idx=np.argsort(-sims)
    return [(id2word[i],round(float(sims[i]),3)) for i in idx[:k] if i!=word2id[w]]
QUAL_PAIRS=[("手机","电脑"),("好","喜欢"),("差","烂"),("便宜","实惠"),("贵","昂贵")]
qual_sg={f"{a}-{b}":round(cos(Wn[word2id[a]],Wn[word2id[b]]),3) for a,b in QUAL_PAIRS if a in word2id and b in word2id}
def cos_w(W_,a,b):
    va=W_[word2id[a]]; vb=W_[word2id[b]]; return cos(va,vb)
qual_cb={f"{a}-{b}":round(cos_w(W_in_cb,a,b),3) for a,b in QUAL_PAIRS if a in word2id and b in word2id}
EXP34={"skipgram_time_s":round(t_sg,2),"cbow_time_s":round(t_cb,2),
       "skipgram_nearest_好":nearest("好"),"cbow_nearest_好":nearest("好") and
          [(id2word[i],round(float(cos_w(W_in_cb,id2word[i],"好")),3)) for i in np.argsort(-(W_in_cb@(W_in_cb[word2id['好']]/np.linalg.norm(W_in_cb[word2id['好']])))) if i!=word2id['好']][:8],
       "qual_skipgram":qual_sg,"qual_cbow":qual_cb}

# ---------- 数值梯度校验（Skip-gram 负采样，小模型）----------
print("[EXP-3] 数值梯度校验 Skip-gram ...")
def sg_grad(W_in,W_out,c,o,negs):
    v=W_in[c].copy(); z_o=W_out[o]@v; s_o=sigmoid(z_o)
    Z_n=W_out[negs]@v; s_n=sigmoid(Z_n)
    L=-np.log(s_o)-np.sum(np.log(sigmoid(-Z_n)))
    gv=(s_o-1.0)*W_out[o]+(s_n@W_out[negs])
    guo=(s_o-1.0)*v; gun=s_n[:,None]*v[None,:]
    return L,gv,guo,gun
sV,sd=12,3
sW_in=np.random.randn(sV,sd)*0.1; sW_out=np.random.randn(sV,sd)*0.1
sent=[0,3,1,5,2,7]; c=4; o=3; negs=np.array([2,8,5,1,9])
gv_num=np.zeros(sd); eps=1e-5
for j in range(sd):
    p=sW_in.copy(); p[c,j]+=eps; m=sW_out.copy()
    Lp=sg_grad(p,m,c,o,negs)[0]
    q=sW_in.copy(); q[c,j]-=eps
    Lq=sg_grad(q,m,c,o,negs)[0]
    gv_num[j]=(Lp-Lq)/(2*eps)
_,gv_ana,_,_=sg_grad(sW_in,sW_out,c,o,negs)
grad_check_sg=float(np.max(np.abs(gv_num-gv_ana))/(np.max(np.abs(gv_num)+np.abs(gv_ana))+1e-12))
print("  Skip-gram 数值梯度最大相对误差 = %.2e"%grad_check_sg)

# =====================================================================
# EXP-2 ：分布假说定量 —— 同一词在不同品类下的上下文向量漂移
# =====================================================================
print("[EXP-2] 计算跨品类上下文漂移 ...")
PROBE=[w for w in ["便宜","好","差","大","小","贵","快","一般"] if w in word2id]
probe_ctx_mean={}
for w in PROBE:
    probe_ctx_mean[w]={}
    for cat in CATEGORIES:
        vecs=[]
        for ids in cat_ids[cat]:
            for i,wid in enumerate(ids):
                if wid==word2id[w]:
                    lo,hi=max(0,i-W2V_WINDOW),min(len(ids),i+W2V_WINDOW+1)
                    ctx=[ids[j] for j in range(lo,hi) if j!=i]
                    if ctx: vecs.append(Wn[ctx].mean(axis=0))
        if len(vecs)>=3: probe_ctx_mean[w][cat]=np.mean(vecs,axis=0)
drift_sim={}; drift_table={}; drift_avg={}
for w in PROBE:
    cats=[c for c in CATEGORIES if c in probe_ctx_mean[w]]
    if len(cats)<2: continue
    M=np.zeros((len(cats),len(cats)))
    for a,ca in enumerate(cats):
        for b,cb in enumerate(cats):
            M[a,b]=cos(probe_ctx_mean[w][ca],probe_ctx_mean[w][cb])
    drift_sim[w]={"cats":cats,"matrix":M.round(3).tolist()}
    nn={}
    for ca in cats:
        q=probe_ctx_mean[w][ca]; sims=Wn@q; order=np.argsort(-sims)
        nn[ca]=[(id2word[i],round(float(sims[i]),3)) for i in order[:3] if i!=word2id[w]]
    drift_table[w]=nn
    drift_avg[w]=float(np.mean(M))
EXP2={"probe_words":PROBE,"drift_similarity":drift_sim,"drift_nearest":drift_table,
      "cross_cat_avg_cos":{w:round(drift_avg[w],3) for w in drift_avg}}

# =====================================================================
# EXP-5 ：负采样 vs 全 softmax（单步耗时 + 降维词表质量对照）
# =====================================================================
print("[EXP-5] 负采样 vs 全 softmax ...")
def time_one_step(kind,V,d,K):
    v=np.random.randn(d)*0.1
    if kind=="neg":
        uo=np.random.randn(d)*0.1; un=np.random.randn(K,d)*0.1; t0=time.time()
        for _ in range(300):
            s_o=sigmoid(uo@v); s_n=sigmoid(un@v)
            (s_o-1)*uo+(s_n@un); (s_o-1)*v; s_n[:,None]*v[None,:]
        return (time.time()-t0)/300
    else:
        U=np.random.randn(V,d)*0.1; t0=time.time()
        for _ in range(20):
            z=U@v; p=sigmoid(z); p/=p.sum(); g=(p-1)[:,None]*v[None,:]
        return (time.time()-t0)/20
t_neg5=time_one_step("neg",V,W2V_DIM,5)
t_neg15=time_one_step("neg",V,W2V_DIM,15)
t_full=time_one_step("full",V,W2V_DIM,0)
print("  单步 negK5=%.2fus negK15=%.2fus fullV=%d=%.2fms"%(t_neg5*1e6,t_neg15*1e6,V,t_full*1e3))

# 降维词表(1500)质量对照：neg-sample vs 全 softmax 各训 1 轮，比近义对余弦
RV=1500
rv_words=vocab[:RV]; rv_id={w:i for i,w in enumerate(rv_words)}
def to_ids_rv(ts): return [rv_id[t] for t in ts if t in rv_id]
# 直接用小语料
rv_corpus=[]
for cat in CATEGORIES:
    for lab,rev in per_cat_reviews[cat][:per_cat_n]:
        ids=to_ids_rv(tok(rev))
        if len(ids)>=2: rv_corpus.append(ids)
def train_full_softmax_rv(corpus,V,d,window,epochs,lr):
    W_in,W_out=init_params(V,d)
    for ep in range(epochs):
        for ids in corpus:
            L=len(ids)
            for i in range(L):
                c=ids[i]; v=W_in[c]
                lo,hi=max(0,i-window),min(L,i+window+1)
                for j in range(lo,hi):
                    if j==i: continue
                    o=ids[j]; z=W_out@v; p=sigmoid(z); p/=p.sum()
                    gv=W_out[o]-(p@W_out)
                    W_in[c]-=lr*gv; W_out[o]-=lr*(p[o]-1)*v
    return W_in
noise_rv=noise[:RV].copy(); noise_rv/=noise_rv.sum()
NEG_TABLE_RV=np.random.choice(RV,size=2_000_000,p=noise_rv)
W_ns=train_skipgram(rv_corpus,RV,W2V_DIM,W2V_WINDOW,W2V_K,1,W2V_LR)[0]
W_fs=train_full_softmax_rv(rv_corpus,RV,W2V_DIM,W2V_WINDOW,1,W2V_LR)
qp=np.mean([cos(W_ns[rv_id[a]]/np.linalg.norm(W_ns[rv_id[a]]),W_ns[rv_id[b]]/np.linalg.norm(W_ns[rv_id[b]])) for a,b in QUAL_PAIRS if a in rv_id and b in rv_id])
qf=np.mean([cos(W_fs[rv_id[a]]/np.linalg.norm(W_fs[rv_id[a]]),W_fs[rv_id[b]]/np.linalg.norm(W_fs[rv_id[b]])) for a,b in QUAL_PAIRS if a in rv_id and b in rv_id])
EXP5={"single_step_us_negK5":round(t_neg5*1e6,2),"single_step_us_negK15":round(t_neg15*1e6,2),
      "single_step_ms_fullV":round(t_full*1e3,3),"vocab_V":V,"reduced_vocab":RV,
      "qual_cos_neg_sample":round(float(qp),3),"qual_cos_full_softmax":round(float(qf),3)}

# =====================================================================
# EXP-6 ：词向量 2D PCA 投影
# =====================================================================
print("[EXP-6] PCA 2D 投影 ...")
from numpy.linalg import svd
topn=300; X=Wn[:topn]; Xc=X-X.mean(axis=0)
U,S,Vt=svd(Xc,full_matrices=False); XY=Xc@Vt.T[:,:2]
EXP6={"top_words":vocab[:60],"xy":XY.tolist()}

# =====================================================================
# EXP-7/8 ：RNN（含 BPTT 数值校验）情感分类 + 时间感受野
# =====================================================================
print("[EXP-7/8] RNN 情感分类与探针 ...")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sent_pairs=[(to_ids(tok(rev)),lab) for cat,lab,rev in rows]
sent_pairs=[(ids,lab) for ids,lab in sent_pairs if 4<=len(ids)<=60]
pos_idx=[i for i,(ids,lab) in enumerate(sent_pairs) if lab==1]
neg_idx=[i for i,(ids,lab) in enumerate(sent_pairs) if lab==0]
np.random.shuffle(pos_idx); np.random.shuffle(neg_idx)
sel=(pos_idx[:N_SENT//2]+neg_idx[:N_SENT//2]); np.random.shuffle(sel)
DATA_SENT=[sent_pairs[i] for i in sel]; print("  情感样本 =",len(DATA_SENT))
X_tr,X_te,y_tr,y_te=train_test_split([ids for ids,_ in DATA_SENT],[lab for _,lab in DATA_SENT],
    test_size=0.2,random_state=SEED,stratify=[lab for _,lab in DATA_SENT])
def bow_vec(ids):
    vs=Wn[ids]; return vs.mean(axis=0) if len(vs) else np.zeros(W2V_DIM)
Xb_tr=np.array([bow_vec(ids) for ids in X_tr]); Xb_te=np.array([bow_vec(ids) for ids in X_te])
lr=LogisticRegression(max_iter=400); lr.fit(Xb_tr,y_tr)
bow_acc=accuracy_score(y_te,lr.predict(Xb_te)); print("  词袋基线准确率 = %.3f"%bow_acc)

def rnn_fb(W_xh,W_hh,b_h,W_hy,b_y,emb,y,skip=False):
    T=len(emb); d=emb.shape[1]; hidden=W_xh.shape[0]
    H=np.zeros((T,hidden)); Z=np.zeros((T,hidden)); h=np.zeros(hidden)
    for t in range(T):
        z=W_xh@emb[t]+W_hh@h+b_h; Z[t]=z
        h=h+np.tanh(z) if skip else np.tanh(z); H[t]=h
    hT=H[-1]; py=sigmoid(W_hy@hT+b_y)[0]
    L=-(y*np.log(py+1e-12)+(1-y)*np.log(1-py+1e-12))
    dl=py-y
    gW_hy=dl*hT; gb_y=dl; gh=W_hy[0]*dl
    gW_xh=np.zeros((hidden,d)); gW_hh=np.zeros((hidden,hidden)); gb_h=np.zeros(hidden)
    for t in reversed(range(T)):
        if skip:
            dz=gh*(1-np.tanh(Z[t])**2)
            gW_xh+=np.outer(dz,emb[t]); gW_hh+=np.outer(dz,H[t-1] if t>0 else 0); gb_h+=dz
            if t>0: gh=gh+W_hh.T@dz
        else:
            dt=gh*(1-H[t]**2)
            gW_xh+=np.outer(dt,emb[t]); gW_hh+=np.outer(dt,H[t-1] if t>0 else 0); gb_h+=dt
            if t>0: gh=W_hh.T@dt
    return L,dict(gW_xh=gW_xh,gW_hh=gW_hh,gb_h=gb_h,gW_hy=gW_hy,gb_y=gb_y)

def train_rnn(X,y,hidden,epochs,lr,skip=False):
    d=W2V_DIM
    W_xh=np.random.randn(hidden,d)*0.1; W_hh=np.random.randn(hidden,hidden)*0.1; b_h=np.zeros(hidden)
    W_hy=np.random.randn(1,hidden)*0.1; b_y=np.zeros(1)
    n=len(X)
    for ep in range(epochs):
        order=np.arange(n); np.random.shuffle(order)
        for ii in order:
            ids=X[ii]; y_i=int(y[ii]); T=len(ids)
            if T==0: continue
            emb=Wn[ids]; _,g=rnn_fb(W_xh,W_hh,b_h,W_hy,b_y,emb,y_i,skip)
            W_xh-=lr*g["gW_xh"]; W_hh-=lr*g["gW_hh"]; b_h-=lr*g["gb_h"]
            W_hy-=lr*g["gW_hy"]; b_y-=lr*g["gb_y"]
    return dict(W_xh=W_xh,W_hh=W_hh,b_h=b_h,W_hy=W_hy,b_y=b_y)
def rnn_predict(p,ids,skip=False):
    d=W2V_DIM; hidden=p["W_xh"].shape[0]; emb=Wn[ids]; T=len(ids)
    if T==0: return 0.5
    h=np.zeros(hidden)
    for t in range(T):
        z=p["W_xh"]@emb[t]+p["W_hh"]@h+p["b_h"]; h=h+np.tanh(z) if skip else np.tanh(z)
    return sigmoid(p["W_hy"]@h+p["b_y"])[0]

rnn_p=train_rnn(X_tr,y_tr,RNN_HID,RNN_EPOCHS,RNN_LR,skip=False)
rnn_acc=accuracy_score(y_te,[1 if rnn_predict(rnn_p,ids)<0.5 else 0 for ids in X_te])
print("  RNN 准确率 = %.3f (词袋基线 %.3f)"%(rnn_acc,bow_acc))

# RNN 数值梯度校验（小模型）
print("[EXP-7] RNN 数值梯度校验 ...")
tiny_ids=[word2id[w] for w in ["东西","很","差","物流","慢","包装"] if w in word2id][:4]
rd,rh=W2V_DIM,5
rW_xh=np.random.randn(rh,rd)*0.5; rW_hh=np.random.randn(rh,rh)*0.5; rb_h=np.zeros(rh)
rW_hy=np.random.randn(1,rh)*0.5; rb_y=np.zeros(1)
emb=Wn[tiny_ids]; yv=1.0
_,ga=rnn_fb(rW_xh,rW_hh,rb_h,rW_hy,rb_y,emb,yv,skip=False)
gWxh_num=np.zeros((rh,rd)); eps=1e-4
for a in range(rh):
    for b in range(rd):
        p1=rW_xh.copy(); p1[a,b]+=eps
        p2=rW_xh.copy(); p2[a,b]-=eps
        Lp=rnn_fb(p1,rW_hh,rb_h,rW_hy,rb_y,emb,yv,skip=False)[0]
        Lq=rnn_fb(p2,rW_hh,rb_h,rW_hy,rb_y,emb,yv,skip=False)[0]
        gWxh_num[a,b]=(Lp-Lq)/(2*eps)
grad_check_rnn=float(np.max(np.abs(gWxh_num-ga["gW_xh"]))/(np.max(np.abs(gWxh_num)+np.abs(ga["gW_xh"]))+1e-12))
print("  RNN 数值梯度最大相对误差 = %.2e"%grad_check_rnn)

# 残差式(直通路)RNN 对照 —— 证明给梯度开直通路救得回长距离
skip_p=train_rnn(X_tr,y_tr,RNN_HID,25,0.30,skip=True)
print("  直通路RNN 测试准确率 = %.3f"%accuracy_score(y_te,[1 if rnn_predict(skip_p,ids,skip=True)<0.5 else 0 for ids in X_te]))

FILL=[w for w in ["这个","商品","收到","包装","感觉","东西","一个","没有","还是","比较","一下","时候"] if w in word2id]
POS_W,NEG_W="喜欢","讨厌"
def build_probe(dist,length,pol):
    seq=[np.random.choice([word2id[w] for w in FILL]) for _ in range(length)]
    seq[length-dist]=word2id[POS_W if pol>0 else NEG_W]; return seq
distances=[1,3,6,10,15]
N_TRIAL=30
rnn_probe={}; skip_probe={}; bow_probe={}
for d_ in distances:
    ar=[]; as_=[]; ab=[]
    for pol in [1,-1]:
        for _ in range(N_TRIAL):
            seq=build_probe(d_,18,pol)
            ar.append(1 if (rnn_predict(rnn_p,seq)<0.5)==(pol<0) else 0)
            as_.append(1 if (rnn_predict(skip_p,seq)<0.5)==(pol<0) else 0)
            ab.append(1 if (lr.predict([bow_vec(seq)])[0]==0)==(pol<0) else 0)
    rnn_probe[d_]=float(np.mean(ar)); skip_probe[d_]=float(np.mean(as_)); bow_probe[d_]=float(np.mean(ab))
print("  RNN 线索距离->准确率:",{d_:round(v,3) for d_,v in rnn_probe.items()})
print("  直通路RNN 线索距离->准确率:",{d_:round(v,3) for d_,v in skip_probe.items()})
print("  词袋 线索距离->准确率:",{d_:round(v,3) for d_,v in bow_probe.items()})

def bptt_grad_norm_decay(p,length):
    ids=[np.random.choice([word2id[w] for w in FILL]) for _ in range(length)]
    emb=Wn[ids]; hidden=p["W_xh"].shape[0]; H=np.zeros((length,hidden)); h=np.zeros(hidden)
    for t in range(length):
        z=p["W_xh"]@emb[t]+p["W_hh"]@h+p["b_h"]; h=np.tanh(z); H[t]=h
    norms=[]
    for k in range(length-1,-1,-1):
        eps=1e-3; Hp=H.copy(); Hp[k]=H[k]+eps; h2=Hp[k].copy()
        for t in range(k+1,length):
            h2=np.tanh(p["W_xh"]@emb[t]+p["W_hh"]@h2+p["b_h"])
        norms.append(float(np.linalg.norm(H[length-1]-h2)/eps))
    return norms
norms=bptt_grad_norm_decay(rnn_p,16); steps_back=list(range(1,len(norms)+1))
EXP7={"bow_baseline_acc":round(float(bow_acc),3),"rnn_acc":round(float(rnn_acc),3),
      "order_gain":round(float(rnn_acc-bow_acc),3),"n_sent":len(DATA_SENT),
      "rnn_hidden":RNN_HID,"rnn_epochs":RNN_EPOCHS,
      "grad_check_rnn":grad_check_rnn}
EXP8={"probe_distances":distances,
      "rnn_probe_acc":{str(k):round(v,3) for k,v in rnn_probe.items()},
      "skip_rnn_probe_acc":{str(k):round(v,3) for k,v in skip_probe.items()},
      "bow_probe_acc":{str(k):round(float(bow_probe[k]),3) for k in distances},
      "bptt_grad_norm":[round(x,4) for x in norms],"bptt_steps_back":steps_back}

# =====================================================================
# 写 stats.json
# =====================================================================
STATS={"EXP1":EXP1,"EXP2":EXP2,"EXP34":EXP34,"EXP5":EXP5,"EXP6":EXP6,"EXP7":EXP7,"EXP8":EXP8,
       "grad_check_skipgram":grad_check_sg,
       "config":{"max_vocab":MAX_VOCAB,"min_count":MIN_COUNT,"n_train_w2v":N_TRAIN_W2V,
                 "w2v_dim":W2V_DIM,"window":W2V_WINDOW,"K":W2V_K,"n_sent":N_SENT,
                 "rnn_hidden":RNN_HID,"rnn_epochs":RNN_EPOCHS}}
with open(os.path.join(ROOT,"stats.json"),"w",encoding="utf-8") as f:
    json.dump(STATS,f,ensure_ascii=False,indent=2)
print("stats.json 写入完成")

# =====================================================================
# 绘图 fig1..fig8
# =====================================================================
print("[绘图] fig1..fig8")
# fig1
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
cat_counts=[len(cat_ids[c]) for c in CATEGORIES]
ax[0].bar(range(10),cat_counts,color=C[2]); ax[0].set_xticks(range(10))
ax[0].set_xticklabels(CATEGORIES,rotation=45,ha="right",fontsize=8)
ax[0].set_title("10 品类评论数（训练子集）",fontsize=10); ax[0].set_ylabel("条数"); style_ax(ax[0])
ax[1].loglog(rank,freq_arr+1,".",color=C[0],ms=3)
ax[1].set_title("词频 Zipf 定律 (r=%.3f)"%zipf_corr,fontsize=10); style_ax(ax[1])
ax[1].set_xlabel("秩(对数)"); ax[1].set_ylabel("频次(对数)")
savefig("fig1_语料与词频.png")

# fig2
fig,ax=plt.subplots(figsize=(7,4)); w="便宜"
present=[c for c in CATEGORIES if c in drift_table.get(w,{})]
ax.text(0.03,0.92,"分布假说：词义 = 它常和谁一起出现",fontsize=11,fontweight="normal")
for i,t in enumerate(drift_table[w][present[0]][:3]):
    ax.text(0.03,0.72-i*0.16,"「"+w+"」在「"+present[0]+"」附近最像："+t[0]+" (%.2f)"%t[1],fontsize=9)
for i,t in enumerate(drift_table[w][present[1]][:3]):
    ax.text(0.03,0.22-i*0.16,"「"+w+"」在「"+present[1]+"」附近最像："+t[0]+" (%.2f)"%t[1],fontsize=9)
ax.set_xlim(0,1); ax.set_ylim(-0.05,1); ax.axis("off")
ax.set_title("同一个词，上下文不同→含义漂移（以「便宜」为例）",fontsize=10)
savefig("fig2_分布假说示意.png")

# fig3
fig,ax=plt.subplots(figsize=(6.5,5.5))
w="好" if "好" in drift_sim else list(drift_sim.keys())[0]
D=drift_sim[w]; M=np.array(D["matrix"]); cats=D["cats"]
im=ax.imshow(M,cmap="YlOrRd",vmin=0.4,vmax=1.0)
ax.set_xticks(range(len(cats))); ax.set_yticks(range(len(cats)))
ax.set_xticklabels(cats,rotation=45,ha="right",fontsize=8); ax.set_yticklabels(cats,fontsize=8)
for i in range(len(cats)):
    for j in range(len(cats)):
        ax.text(j,i,"%.2f"%M[i,j],ha="center",va="center",fontsize=7,color="black")
ax.set_title("「%s」跨 10 品类的上下文向量余弦相似度"%w,fontsize=10)
fig.colorbar(im,fraction=0.046,pad=0.04)
savefig("fig3_跨品类漂移热力图.png")

# fig4
fig,ax=plt.subplots(1,2,figsize=(9,3.8))
ax[0].text(0.5,0.78,"CBOW：多对一\n上下文 → 中心词",ha="center",fontsize=10)
ax[0].text(0.5,0.35,"Skip-gram：一对多\n中心词 → 上下文",ha="center",fontsize=10)
ax[0].axis("off"); ax[0].set_title("两种训练方式",fontsize=10)
bars=ax[1].bar(["Skip-gram","CBOW"],[t_sg,t_cb],color=[C[4],C[6]])
ax[1].set_title("单轮训练耗时(秒)",fontsize=10); style_ax(ax[1])
for b,v in zip(bars,[t_sg,t_cb]): ax[1].text(b.get_x()+b.get_width()/2,v,"%.1f"%v,ha="center",va="bottom",fontsize=9)
savefig("fig4_CBOW_Skipgram.png")

# fig5
fig,ax=plt.subplots(1,2,figsize=(9,3.8))
ax[0].text(0.5,0.7,"真词 + K 个噪声词\n→ K+1 个二分类(sigmoid)\n复杂度 O(K)",ha="center",fontsize=10)
ax[0].text(0.5,0.22,"全 softmax：分母遍历\n整个词表 复杂度 O(V)=O(%d)"%V,ha="center",fontsize=10)
ax[0].axis("off"); ax[0].set_title("负采样 vs 全 softmax",fontsize=10)
xs=["neg K=5","neg K=15","全 softmax V=%d"%V]; ys=[t_neg5*1e6,t_neg15*1e6,t_full*1e3]
bars=ax[1].bar(xs,ys,color=[C[1],C[3],C[5]]); ax[1].set_yscale("log")
ax[1].set_ylabel("单步耗时(μs / ms)"); style_ax(ax[1]); ax[1].set_title("算得动是设计出来的",fontsize=10)
savefig("fig5_负采样对照.png")

# fig6
fig,ax=plt.subplots(figsize=(7,6))
ax.scatter(XY[:,0],XY[:,1],s=12,color=C[8],alpha=0.6)
for w in ["手机","电脑","好","差","便宜","贵","喜欢","讨厌"]:
    if w in word2id and word2id[w]<topn:
        i=word2id[w]; ax.annotate(w,XY[i],fontsize=9,color=C[0])
ax.set_title("词向量 PCA 二维投影（语义聚类）",fontsize=10); style_ax(ax); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
savefig("fig6_词向量PCA.png")

# fig7
fig,ax=plt.subplots(1,2,figsize=(9,3.8))
ax[0].text(0.05,0.6,"h_t = tanh(W·x_t + U·h_{t-1} + b)\n同一矩阵 U 每个时间步复用\n沿时间展开 = BPTT",fontsize=9,va="center")
ax[0].axis("off"); ax[0].set_title("RNN 沿时间展开",fontsize=10)
ax[1].plot(steps_back,norms,"-o",color=C[5],ms=4)
ax[1].set_title("BPTT 梯度范数随回溯步数衰减",fontsize=10); style_ax(ax[1])
ax[1].set_xlabel("回溯步数 T-k"); ax[1].set_ylabel("||∂h_T/∂h_k||")
savefig("fig7_RNN与梯度衰减.png")

# fig8
fig,ax=plt.subplots(figsize=(7,4))
xs=distances
ax.plot(xs,[rnn_probe[d] for d in xs],"-o",color=C[5],label="RNN(顺序)")
ax.plot(xs,[skip_probe[d] for d in xs],"-^",color=C[3],label="直通路RNN(门控雏形)")
ax.plot(xs,[bow_probe[d] for d in xs],"--s",color=C[8],label="词袋(无顺序)")
ax.set_title("时间感受野：情感线索距句尾越远，RNN 越抓不住",fontsize=10); style_ax(ax)
ax.set_xlabel("线索距句尾的步数"); ax.set_ylabel("准确率"); ax.legend(fontsize=8)
savefig("fig8_时间感受野.png")

print("全部完成。")
