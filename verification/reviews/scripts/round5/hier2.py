import numpy as np, sys
def lse(a,axis):
    m=np.max(a,axis=axis,keepdims=True)
    m=np.where(np.isfinite(m),m,0)
    return (m+np.log(np.exp(a-m).sum(axis=axis,keepdims=True))).squeeze(axis)
def run(G,h,g,qs):
    xs=(np.arange(G)+0.5)/G; gs=g(xs)
    nodes=[]
    def rec(lo,hi,par):
        i=len(nodes); nodes.append((lo,hi,par))
        if hi-lo>1:
            mid=(lo+hi)//2; rec(lo,mid,i); rec(mid,hi,i)
    rec(0,G,-1)
    n=len(nodes)
    lo=np.array([a for a,b,p in nodes]); hi=np.array([b for a,b,p in nodes]); par=np.array([p for a,b,p in nodes])
    cnt=(hi-lo).astype(float); mu=np.array([xs[a:b].mean() for a,b,p in nodes]); var=np.array([xs[a:b].var() for a,b,p in nodes])+h*h
    gy=np.array([gs[a:b].mean() for a,b,p in nodes]); leaf=cnt==1
    Q=qs[:,None]
    ll=-0.5*np.log(2*np.pi*var)-0.5*(Q-mu)**2/var
    pi=np.zeros_like(ll)
    un=ll+np.log(cnt/np.where(par>=0,cnt[np.maximum(par,0)],1))
    children={}
    for i in range(1,n): children.setdefault(par[i],[]).append(i)
    for p in range(n):
        if p in children:
            sib=children[p]; z=lse(un[:,sib],axis=1)
            for i in sib: pi[:,i]=pi[:,p]+un[:,i]-z
    res={}
    for wt in ['lik','prior','post','path']:
        if wt=='lik': s=ll.copy()
        elif wt=='prior': s=ll+np.log(cnt/G)
        elif wt=='post': s=pi.copy(); s[:,0]=ll[:,0]
        else: s=pi+ll; s[:,0]=ll[:,0]
        for lo_ in [False,True]:
            ss=s.copy()
            if lo_: ss[:,~leaf]=-np.inf
            w=np.exp(ss-lse(ss,axis=1)[:,None]); pred=(w*gy).sum(1)
            res[(wt,lo_)]=np.sqrt(np.mean((pred-g(qs))**2))
    return res
g=lambda x: np.sin(np.pi*x)
qs=np.random.default_rng(0).uniform(0.05,0.95,300)
for D in [4,6,8,10,12]:
    G=2**D; r=run(G,1.0/G,g,qs)
    print(D,' '.join(f"{k[0]}{'L' if k[1] else 'A'}={v:.5f}" for k,v in r.items()),flush=True)
