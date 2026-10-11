import numpy as np
# sup_x sum_g w_g ||g-x|| / delta for 2D cell-centre grid, h = delta/2 (as in R2), interior and boundary
m=16; d=1.0/m; h=d/2
g=(np.arange(m)+0.5)*d; G=np.array(np.meshgrid(g,g)).reshape(2,-1).T
xs=np.random.default_rng(0).uniform(0,1,(20000,2))
worst=0
for x in xs:
    r2=((G-x)**2).sum(1); a=np.exp(-(r2-r2.min())/(2*h*h)); w=a/a.sum()
    worst=max(worst,(w*np.sqrt(r2)).sum()/d)
print("C (error <= L_euclid * C * spacing):",worst)
L=1+np.pi/2*2*1.25*1.1
for D in (1,2,3):
    S=sum(L**l for l in range(D))
    # need C*L*(w/m)*S <= r, r = 0.1*w0, w = 1.2*w0
    m_needed=worst*L*1.2*S/0.1
    print("D",D,"L",round(L,2),"sumL",round(S,1),"m needed >=",round(m_needed))
