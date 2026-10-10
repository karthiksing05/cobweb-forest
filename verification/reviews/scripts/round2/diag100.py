import numpy as np
rng = np.random.default_rng(0)
n, N = 100, 200000
# Y = exp( (1/100) sum_i sin^2(pi x_i / 2) ), x ~ U[-1,1]^100
X = rng.uniform(-1, 1, size=(N, n)).astype(np.float64)
S = np.sin(np.pi * X / 2) ** 2
Y = np.exp(S.mean(axis=1))
sdY = Y.std()
# exact E[Y | X_1] = exp(s_1/100) * E[exp(S_{-1}/100)], E over the other 99 coords
# E[exp(s/100)] for one coord, computed by quadrature
u = np.linspace(-1, 1, 200001)
m1 = np.trapezoid(np.exp(np.sin(np.pi*u/2)**2/100), u) / 2.0
c = m1 ** 99
cond = np.exp(S[:, 0] / 100) * c
print("sd(Y)            =", sdY)
print("sd(E[Y|X_1])     =", cond.std())
print("ratio sd(E[Y|X1])/sd(Y) =", cond.std() / sdY, " 1/ratio =", sdY / cond.std())
print("1/sqrt(100) =", 0.1, "  1/7 =", 1/7)
