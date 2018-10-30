import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib
from scipy.integrate import ode

mux = 1
muy=1
mxx=-1
myy = -1
mxy = 1.5
myx = 1.5
PopIn = (mux,muy,mxx,myy,mxy,myx)
x0 = [0.9,0.1]


def logi(t,y,PopIn):
    dzdt = [y[0]*(PopIn[0]+PopIn[2]*y[0]+PopIn[4]*y[1]),y[1]*(PopIn[1]+PopIn[5]*y[0]+PopIn[3]*y[1])]
    return dzdt



r = ode(logi).set_integrator('vode', method='bdf')
r.set_initial_value(x0, 0)
t1 = 10
dt = 1
while r.successful() and r.t < t1:
     print(r.t+dt, r.integrate(r.t+dt))
    
plt.legend(loc='best')
plt.xlabel('t')
plt.title('(Mxy,Myx) = (1.5,1.5),(x0,y0) = (.9,.1)')
plt.grid()
plt.show()