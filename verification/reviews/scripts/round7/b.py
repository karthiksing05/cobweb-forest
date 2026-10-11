import numpy as np
kap=np.array([1759,328,1654,88083,1218,4161,25229,7400,3714.])
err=np.array([.024,.003,.003,.064,.003,.032,.448,.039,.147])
def rank(a):
    o=np.argsort(a,kind='stable'); r=np.empty(len(a)); 
    s=np.sort(a); 
    for i,v in enumerate(a): r[i]=np.mean(np.where(s==v)[0])+1
    return r
rk,re=rank(kap),rank(err); print(rk,re)
print("spearman",np.corrcoef(rk,re)[0,1])
# with untied hypothetical (different precision)
