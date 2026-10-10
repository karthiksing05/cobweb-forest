import numpy as np
N = np.array([100, 400, 1600, 6400])
w = {'sin prior': [0.197, 0.166, 0.132, 0.112], 'j0 prior': [0.158, 0.150, 0.129, 0.107],
     'sin post': [0.113, 0.135, 0.085, 0.081], 'j0 post': [0.175, 0.180, 0.183, 0.162],
     'sin leaves': [0.019, 0.0066, 0.0016, 0.00038]}
for k, v in w.items():
    v = np.array(v)
    slope = np.polyfit(np.log(N), np.log(v), 1)[0]
    print(f'{k:11s} loglog slope={slope:+.3f}  err*lnN={np.round(v*np.log(N),3)}  per-quadrupling factors={np.round(v[:-1]/v[1:],2)}')
Nt = np.array([250, 1000, 4000, 16000])
toys = {'nested4 single': [0.765, 0.574, 0.393, 0.277], 'nested4 act': [0.618, 0.460, 0.376, 0.326],
        'noisy single': [0.807, 0.609, 0.425, 0.319], 'noisy act': [0.630, 0.468, 0.393, 0.320]}
for k, v in toys.items():
    v = np.array(v); print(f'{k:15s} loglog slope={np.polyfit(np.log(Nt), np.log(v), 1)[0]:+.3f}')
print('forest/single ratio nested4:', np.round(np.array(toys['nested4 act'])/np.array(toys['nested4 single']), 3))
print('forest/single ratio noisy  :', np.round(np.array(toys['noisy act'])/np.array(toys['noisy single']), 3))
print('crossover exponents (interior): n=4 single noise-free 2/(n+2)=%.3f, forest k=2 noisy 2/(k+4)=%.3f; noisy single 2/(n+4)=%.3f' % (2/6, 2/6, 2/8))
# sin/expsin tree vs MLP/KAN
for name, tree, mlp, kan in [('sin', [0.019,0.007,0.0038,0.0025],[0.0025,0.00058,0.0006,0.00036],[0.012,0.00027,0.00024,0.00016]),
                             ('expsin', [0.167,0.079,0.041,0.018],[0.027,0.0088,0.0033,0.0017],[0.065,0.0025,0.0011,0.00053])]:
    print(name, 'tree/MLP', np.round(np.array(tree)/np.array(mlp),1), 'tree/KAN', np.round(np.array(tree)/np.array(kan),1))
# Feynman N4000 best cobweb vs mlp/kan
fey = {'I.12.11':(0.140,0.004,0.002),'I.13.12':(0.281,0.006,0.005),'I.29.16':(0.182,0.011,0.095),'I.37.4':(0.092,0.003,0.003),
       'I.50.26':(0.308,0.016,0.125),'I.6.20':(0.021,0.001,0.001),'I.6.20b':(0.078,0.003,0.020),'I.9.18':(0.322,0.003,0.015),'II.6.15a':(0.421,0.013,0.024)}
f = {k:(round(c/m,1), round(c/kk,1)) for k,(c,m,kk) in fey.items()}
print('best-cobweb / (MLP, KAN):', f)
# selection within 1%
for s, b in [(0.662,0.657),(0.451,0.450),(0.334,0.334)]: print('sel/best', round(s/b,4))
for s, b in [(0.327,0.326),(0.546,0.494)]: print('induce sel/best', round(s/b,4))
