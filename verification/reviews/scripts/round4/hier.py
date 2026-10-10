import numpy as np
# balanced binary tree over 1-D grid of G=2^D cell centres on [0,1], instances (x, g(x)), query Y missing
def run(D, g, beta=1.0):
    G=2**D; d=1.0/G; xs=(np.arange(G)+0.5)*d; ys=g(xs); h=d/beta; h2=h*h
    # nodes by level: level k has 2^k nodes, each pooling G/2^k consecutive leaves
    levels=[]
    for k in range(D+1):
        m=2**k; blk=G//m
        X=xs.reshape(m,blk); Y=ys.reshape(m,blk)
        levels.append((X.mean(1), X.var(1)+h2, Y.mean(1), np.full(m,blk/G)))
    q=np.linspace(0.1,0.9,801)  # interior
    def ll(mu,v): return -0.5*(np.log(2*np.pi*v)[None,:]+(q[:,None]-mu[None,:])**2/v[None,:])
    LL=[ll(mu,v) for (mu,v,_,_) in levels]
    # path posteriors pi
    PI=[np.zeros((len(q),1))]
    for k in range(1,D+1):
        mu,v,_,pr=levels[k]; par=PI[k-1]
        u=LL[k]+np.log(0.5)   # count ratio child/parent=1/2
        # siblings pairs (2i,2i+1)
        u2=u.reshape(len(q),-1,2); lse=np.logaddexp(u2[:,:,0],u2[:,:,1])
        PI.append((u2-lse[:,:,None]).reshape(len(q),-1)+np.repeat(par,2,axis=1))
    out={}
    def mix(scores, means):
        S=np.concatenate(scores,1); M=np.concatenate(means)
        S=S-S.max(1,keepdims=True); W=np.exp(S); W/=W.sum(1,keepdims=True)
        return W@M
    f=g(q)
    allY=[l[2] for l in levels]
    s_prior=[LL[k]+np.log(levels[k][3])[None,:] for k in range(D+1)]
    s_lik=[LL[k] for k in range(D+1)]
    s_post=[LL[0]]+[PI[k] for k in range(1,D+1)]
    s_path=[LL[0]]+[PI[k]+LL[k] for k in range(1,D+1)]
    for name,s in [('prior_all',s_prior),('lik_all',s_lik),('post_all',s_post),('path_all',s_path)]:
        out[name]=np.sqrt(np.mean((mix(s,allY)-f)**2))
    for name,s in [('post_leaves',s_post[D]),('lik_leaves',s_lik[D]),('path_leaves',s_path[D])]:
        out[name]=np.sqrt(np.mean((mix([s],[allY[D]])-f)**2))
    return out
for gname,g in [('sin',lambda x: np.sin(np.pi*(2*x-1))),('j0like',lambda x: np.cos(20*(2*x-1)))]:
    print(gname)
    for D in [4,6,8,10,12,14]:
        o=run(D,g); print(D, ' '.join(f'{k}={v:.4g}' for k,v in o.items()), ' prior*D=%.3f'%(o['prior_all']*D))
