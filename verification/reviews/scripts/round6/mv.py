import numpy as np
# latent Z (1), E1 (2), E2 (2): standard basis vector w = (Z, e11, e12, e21, e22) standard normals
L1=np.array([1,0.8]); L2=np.array([0.9,-0.7]); s1=np.array([0.5,0.6]); s2=np.array([0.6,0.5])
def row(i): v=np.zeros(5); v[i]=1; return v
X1=np.stack([L1[0]*row(0)+s1[0]*row(1), L1[1]*row(0)+s1[1]*row(2)])
X2=np.stack([L2[0]*row(0)+s2[0]*row(3), L2[1]*row(0)+s2[1]*row(4)])
Y=s1[0]*row(1)  # E_{1,1} with scale 0.5
def proj(target, feats):  # L2 projection coefficient vector of target onto span(feats)
    A=feats; G=A@A.T; b=A@target; c=np.linalg.solve(G,b); return c@A
X=np.vstack([X1,X2])
f=proj(Y,X)
mu1=proj(row(0),X1); mu2=proj(row(0),X2)
F=proj(f,np.vstack([mu1,mu2]))
r=np.sum((f-F)**2)/np.sum(f**2)
print("Delta/Varf",r,"sqrt",np.sqrt(r))
# variant: Y = E_{1,1} unscaled (unit)
