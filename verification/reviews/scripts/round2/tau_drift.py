# is_exact_match (continuous part): accept x into a leaf iff  var = sum_sq/count <= 1e-5 (all dims)
# and |x - mean| <= 1e-5 (all dims).  Welford update in float32 as in increment_counts.
# (is_close_to_zero is in helper.h, not shown; we assume max-abs <= tol.)
import numpy as np
tol = np.float32(1e-5)
mean = np.float32(0.0); ss = np.float32(0.0); n = 0; xs = []
def accept(x):
    global mean, ss, n
    if n > 0 and (ss / np.float32(n) > tol or abs(x - mean) > tol): return False
    n += 1; d = x - mean; mean = mean + d / np.float32(n); ss = ss + d * (x - mean); xs.append(x); return True
accept(np.float32(0.0))
for k in range(2000):
    x = np.float32(mean + np.float32(0.99e-5))
    if not accept(x): break
xs = np.array(xs, dtype=np.float64)
print("copies stored in ONE leaf:", n)
print("max |x_i - leaf mean| =", np.abs(xs - float(mean)).max(), " = %.2f tau" % (np.abs(xs - float(mean)).max() / 1e-5))
print("spread (max - min)    =", xs.max() - xs.min(), " = %.2f tau" % ((xs.max() - xs.min()) / 1e-5))
