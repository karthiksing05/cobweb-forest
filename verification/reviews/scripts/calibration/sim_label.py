import numpy as np, itertools, math
# thm:label construction: n label trees with K grid labels; output tree = one leaf per tuple t (count 1), y=f(centre)
# LIK leaves-only: leaf score = sum_A log P_c(A=q_A), P_c = (n+alpha)/(1+(K+1)alpha) (code: log_prob_instance)
def pred(f, x, n, K, alpha):
    t = tuple(min(int(xi * K), K - 1) for xi in x)
    num = den = 0.0
    for tp in itertools.product(range(K), repeat=n):
        m = sum(a != b for a, b in zip(t, tp))
        w = ((alpha) / (1 + alpha)) ** m          # ratio relative to the matching leaf
        c = [(j + 0.5) / K for j in tp]
        num += w * f(c); den += w
    return num / den
f = lambda c: c[0]
for (n, K, alpha) in [(1, 100, 0.1), (1, 100, 0.01), (2, 20, 0.05)]:
    errs = []
    for x in [(0.005,)*n, (0.5,)*n, (0.995,)*n]:
        errs.append(abs(pred(lambda c: c[0], x, n, K, alpha) - x[0]))
    paper = math.sqrt(n) / (2 * K) + 2 * 1.0 * ((1 + alpha) ** n - 1)   # omega_f(sqrt(n) delta/2) + 2||f||((1+a)^n-1)
    corrected = math.sqrt(n) / (2 * K) + 2 * 1.0 * ((1 + (K - 1) * alpha / (1 + alpha)) ** n - 1)
    print(f"n={n} K={K} alpha={alpha}: max error {max(errs):.4f} | paper's bound {paper:.4f} | corrected bound {corrected:.4f}")
