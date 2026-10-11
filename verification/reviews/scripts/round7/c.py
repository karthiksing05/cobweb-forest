import numpy as np
rng=np.random.default_rng(0)
N=2_000_000
X=rng.uniform(-1,1,(N,6))
# pairs (0,1),(2,3) with c=1.5,0.8 ; monotone x4 -> x^3, x5 -> tanh(2x)
c=[1.5,0.8]
Y=c[0]*X[:,0]*X[:,1]+c[1]*X[:,2]*X[:,3]+X[:,4]**3+np.tanh(2*X[:,5])+0.1*rng.standard_normal(N)
B=40
def condmean_var(cols):
    # Var of E[Y|cols] via binning (cols list of arrays)
    idx=np.zeros(N,dtype=np.int64)
    for a in cols:
        q=np.clip(((a-a.min())/(a.max()-a.min()+1e-12)*B).astype(int),0,B-1)
        idx=idx*B+q
    s=np.bincount(idx,Y); n=np.bincount(idx)
    m=np.where(n>0,s/np.maximum(n,1),0)
    return np.var(m[idx])
def act(j):
    # E[Y|x_j] by binning in 1D
    q=np.clip(((X[:,j]+1)/2*200).astype(int),0,199)
    s=np.bincount(q,Y,200); n=np.bincount(q,minlength=200)
    return (s/n)[q]
acts=[act(j) for j in range(6)]
for i in range(6):
    for j in range(i+1,6):
        vfull=condmean_var([X[:,i],X[:,j]])
        a1,a2=acts[i],acts[j]
        # if activation (near-)constant, condition on it = condition on nothing
        cols=[a for a in (a1,a2) if a.std()>0.02]
        vact=condmean_var(cols) if cols else 0.0
        print(i,j,"deficit~%.4f"%(vfull-vact), "act sd %.3f %.3f"%(a1.std(),a2.std()))
print("c^2/9:",[ci**2/9 for ci in c])
