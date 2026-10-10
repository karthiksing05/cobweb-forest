"""Two-view Gaussian model: Z~N(0,1) scalar, X^(v)=lam_v Z + E_v (2-D), E_v~N(0, s^2 I).
Target Y = E_{1,1} (first noise coord of view 1).  f = E[Y|X] (linear).  Cross-view ideal
messages m_v = Lambda_W mu_v with mu_v = E[Z|X^(v)] (scalar, linear).  F_A = E[f | mu_1, mu_2].
Report sqrt(Delta)/sd(f).  Also verify Prop multiview: E[Z|X] is a function of (mu_1, mu_2)."""
import numpy as np

def analyse(l1, l2, s1, s2):
    l1 = np.array(l1, float); l2 = np.array(l2, float)
    # joint vector W = (Z, E1(2), E2(2)); X = A W
    # X = [l1 Z + E1 ; l2 Z + E2]
    A = np.zeros((4, 5)); A[0:2, 0] = l1; A[2:4, 0] = l2
    A[0:2, 1:3] = np.eye(2); A[2:4, 3:5] = np.eye(2)
    CW = np.diag([1, s1**2, s1**2, s2**2, s2**2])
    CX = A @ CW @ A.T
    # f = E[E11 | X] = c^T X with c = CX^{-1} Cov(X, E11)
    covXE11 = A @ CW[:, 1]
    c = np.linalg.solve(CX, covXE11)
    # mu_v = k_v^T X^(v)
    def kv(l, s):
        Cv = np.outer(l, l) + s**2 * np.eye(2)
        return np.linalg.solve(Cv, l)        # Cov(X^(v), Z) = l
    k1 = np.zeros(4); k1[0:2] = kv(l1, s1)
    k2 = np.zeros(4); k2[2:4] = kv(l2, s2)
    Kmat = np.stack([k1, k2])                 # mu = K X
    # projection of f onto span(mu1, mu2): F_A = b^T mu
    Cmu = Kmat @ CX @ Kmat.T
    b = np.linalg.solve(Cmu, Kmat @ CX @ c)
    varf = c @ CX @ c
    resid = c - Kmat.T @ b
    delta = resid @ CX @ resid
    # check E[Z|X] in span(mu1,mu2)
    covXZ = A @ CW[:, 0]
    z = np.linalg.solve(CX, covXZ)
    bz = np.linalg.lstsq(Kmat.T, z, rcond=None)[0]
    zres = np.linalg.norm(z - Kmat.T @ bz)
    return np.sqrt(delta / varf), zres

for l1, l2, s in [((1, 1), (1, 1), 1.0), ((1, 1), (1, 1), 0.5), ((1, 1), (1, 1), 2.0),
                  ((1, 0.5), (1, 0.5), 1.0), ((2, 1), (1, 2), 1.0), ((1, 1), (1, 1), 0.3)]:
    r, zres = analyse(l1, l2, s, s)
    print(f'lam1={l1} lam2={l2} noise sd={s}: sqrt(Delta)/sd(f) = {r:.3f};  ||E[Z|X] - span(mu)|| = {zres:.1e}')
