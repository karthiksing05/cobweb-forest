import numpy as np
def logsumexp(a):
    a=np.asarray(a); m=a.max(); return m+np.log(np.exp(a-m).sum())
def build(G):
    # balanced binary tree over grid points; return nodes with (count, mean, var_emp, children)
    xs=(np.arange(G)+0.5)/G
    return xs
def predict(xs,gs,h,q,weighting,leaves_only):
    G=len(xs)
    # nodes: intervals [lo,hi) recursively
    out_s=[];out_mu=[]
    def ll(lo,hi):
        pts=xs[lo:hi]; m=pts.mean(); v=pts.var()+h*h
        return -0.5*np.log(2*np.pi*v)-0.5*(q-m)**2/v
    root_ll=ll(0,G)
    out=[]
    def rec(lo,hi,pi,depth):
        n=hi-lo
        if n==1: return
        mid=(lo+hi)//2
        ch=[(lo,mid),(mid,hi)]
        lls=np.array([ll(a,b) for a,b in ch]); cnt=np.array([b-a for a,b in ch])
        un=lls+np.log(cnt/n); post=un-logsumexp(un)
        for (a,b),l,p in zip(ch,lls,post):
            cpi=pi+p
            if weighting=='lik': s=l
            elif weighting=='prior': s=l+np.log((b-a)/G)
            elif weighting=='post': s=cpi
            else: s=cpi+l
            leaf=(b-a)==1
            if leaf or not leaves_only: out.append((s,gs[a:b].mean()))
            rec(a,b,cpi,depth+1)
    if not leaves_only: out.append((root_ll,gs.mean()))
    rec(0,G,0.0,0)
    s=np.array([o[0] for o in out]); mu=np.array([o[1] for o in out])
    w=np.exp(s-logsumexp(s)); return (w*mu).sum()
g=lambda x: np.sin(np.pi*x)
for D in [4,6,8,10]:
    G=2**D; xs=(np.arange(G)+0.5)/G; gs=g(xs); h=1.0/G
    qs=np.linspace(0.1,0.9,97)
    res={}
    for wt in ['lik','prior','post','path']:
        for lo in [False,True]:
            e=[predict(xs,gs,h,q,wt,lo)-g(q) for q in qs]
            res[(wt,lo)]=np.sqrt(np.mean(np.square(e)))
    print(D, ' '.join(f"{k[0]}{'L' if k[1] else 'A'}={v:.4f}" for k,v in res.items()))
