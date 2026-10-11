import numpy as np
# r=1, two one-dimensional views, view-specific target Y=E_1
def row(i): v=np.zeros(3); v[i]=1; return v
X1=1.0*row(0)+0.5*row(1); X2=0.9*row(0)+0.6*row(2); Y=0.5*row(1)
def proj(t,A):
    A=np.atleast_2d(A); c=np.linalg.solve(A@A.T,A@t); return c@A
f=proj(Y,np.vstack([X1,X2])); mu1=proj(row(0),X1); mu2=proj(row(0),X2)
F=proj(f,np.vstack([mu1,mu2])); print("Delta/Varf (1-dim views, view-specific target):",np.sum((f-F)**2)/np.sum(f**2))
