import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('bf', '/tmp/referee_n7/binary_full.py')
src = open('/tmp/referee_n7/binary_full.py').read().split("g = lambda x")[0]
ns = {}; exec(src, ns)
run = ns['run']
rng = np.random.default_rng(2)
xq = rng.uniform(0.1, 0.9, 200)
for name, g in [('linear 2x-1', lambda x: 2 * x - 1), ('J0-like cos(20x)', lambda x: np.cos(20 * (2 * x - 1)))]:
    print('target', name)
    for D in [6, 10, 14, 16]:
        chunks = np.array_split(xq, max(1, 2 ** D // 4096))
        parts = [run(g, D, c) for c in chunks]
        out = {k: np.sqrt(np.mean([p[k] ** 2 for p in parts])) for k in parts[0]}
        print(f'  D={D:2d}: PRIOR-all {out["prior"]:.4f}  POSTERIOR-all {out["posterior"]:.4f}  POSTERIOR-leaves {out["post_leaves"]:.4f}')
