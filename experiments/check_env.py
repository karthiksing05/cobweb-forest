import hashlib, glob, socket, sys
import numpy as np
print(socket.gethostname(), sys.version)
for p in sorted(glob.glob("cobweb_forest/*.py")):
    print(hashlib.md5(open(p, "rb").read()).hexdigest(), p)
import cobweb_forest, cobweb_forest._core as c
print(cobweb_forest.__file__, c.__file__)
from cobweb_forest import Forest, designs as D
rng = np.random.default_rng(0)
f = lambda X: np.exp(np.sin(np.pi * X[:, 0]) + X[:, 1]**2)
N = 2000
X = rng.uniform(-1, 1, (N, 2)).astype(np.float32); Y = f(X)[:, None].astype(np.float32)
Xt = rng.uniform(-1, 1, (500, 2)).astype(np.float32); Yt = f(Xt)
for name, sp in [("single", D.single(2, N, kappa=0.1, max_nodes=300, leaves_only=True)),
                 ("act", D.ka((0, 1), "act", N, kappa=0.1, max_nodes=300, leaves_only=True))]:
    F = Forest(sp, "out").fit(X, Y, protocol="online"); P = F.predict(Xt)[:, 0]
    print(name, np.sqrt(np.mean((P - Yt) ** 2)), [s.prior_var for s in sp])
