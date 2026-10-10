# f(x1,x2) = x1*x2 + x1,  X1 ~ U[-1,3], X2 ~ U[-1,1] independent, Y = f(X) (noise-free).
# Funnel messages: m1(x1) = E[Y|X1=x1] = x1*(E X2 + 1) = x1 ; m2(x2) = E[Y|X2=x2] = E[X1]*(x2+1) = x2+1.
# Both are injective, so (m1,m2) determines (x1,x2) and F_A = E[Y | m1, m2] = f  -> sufficient.
# f is NOT of the form Phi(g1(x1)+g2(x2)) with Phi strictly increasing: such an f orders x2-values
# the same way for every x1, but here d f / d x2 = x1 changes sign.
import numpy as np
rng = np.random.default_rng(0)
N = 2_000_000
x1 = rng.uniform(-1, 3, N); x2 = rng.uniform(-1, 1, N); y = x1 * x2 + x1
# empirical conditional means on bins
b1 = np.digitize(x1, np.linspace(-1, 3, 41)); b2 = np.digitize(x2, np.linspace(-1, 1, 41))
m1 = np.array([y[b1 == k].mean() for k in range(1, 41)])
m2 = np.array([y[b2 == k].mean() for k in range(1, 41)])
print("E[Y|X1] binned strictly increasing:", np.all(np.diff(m1) > 0))
print("E[Y|X2] binned strictly increasing:", np.all(np.diff(m2) > 0))
f = lambda a, b: a * b + a
print("x1=2  : f(2,-0.5) =", f(2, -0.5), " f(2,0.5) =", f(2, 0.5), "-> increasing in x2")
print("x1=-.5: f(-.5,-0.5) =", f(-.5, -0.5), " f(-.5,0.5) =", f(-.5, 0.5), "-> decreasing in x2")
