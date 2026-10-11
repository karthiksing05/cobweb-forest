import numpy as np
from math import comb
# Spearman
kap=np.array([1759,328,1654,88083,1218,4161,25229,7400,3714.])
err=np.array([.024,.003,.003,.064,.003,.032,.448,.039,.147])
def rank(a):
    from scipy.stats import rankdata
    return rankdata(a)
try:
    from scipy.stats import spearmanr, kendalltau
    print("spearman",spearmanr(kap,err), "kendall",kendalltau(kap,err))
except Exception as e: print(e)
# Bernstein
f=lambda x,y: np.exp(np.sin(np.pi*x)+y**2)
xs=np.linspace(0,1,401); X,Y=np.meshgrid(xs,xs,indexing='ij')
def B(m):
    out=np.zeros_like(X)
    bx=[comb(m,i)*xs**i*(1-xs)**(m-i) for i in range(m+1)]
    for i in range(m+1):
        for j in range(m+1):
            out+=f(i/m,j/m)*np.outer(bx[i],bx[j])
    return out
errs={}
for m in [2,4,6,8,10,12]:
    errs[m]=np.abs(B(m)-f(X,Y)).max(); print(m,errs[m])
print("local slope 10->12",np.log(errs[12]/errs[10])/np.log(12/10))
# cross-view floor
Lam1=np.array([1,0.8]); Lam2=np.array([0.9,-0.7]); s1=np.array([.5,.6]); s2=np.array([.6,.5])
# latent basis: Z, E1(2), E2(2) standard normals
def vec(z,e1,e2): return np.concatenate([[z],e1,e2])
X1=[vec(Lam1[i],np.eye(2)[i]*s1[i],np.zeros(2)) for i in range(2)]
X2=[vec(Lam2[i],np.zeros(2),np.eye(2)[i]*s2[i]) for i in range(2)]
Xs=np.array(X1+X2)  # 4x5
Yv=vec(0,np.array([1,0]),np.zeros(2))
def proj(target, basis):
    G=basis@basis.T; c=np.linalg.solve(G,basis@target); return c@basis
fv=proj(Yv,Xs)
# mu_v = E[Z|X^(v)]
zv=vec(1,np.zeros(2),np.zeros(2))
mu1=proj(zv,np.array(X1)); mu2=proj(zv,np.array(X2))
Fv=proj(Yv,np.array([mu1,mu2]))
print("floor",np.sqrt(((fv-Fv)**2).sum()/(fv**2).sum()))
