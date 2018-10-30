import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib
from scipy import integrate
from scipy.integrate import odeint

#x'(t) = x(t)*(mu-M*x(t))
def logi(z,t,mux,muy,mxx,mxy,myx,myy):
    x,y = z
    dzdt = [x*(mux+mxx*x+mxy*y),y*(muy+myx*x+myy*y)]
    return dzdt

mux = 1
muy=1
mxx=-1
myy = -1
mxy = -.5
myx = -.5
x0 = [0.1,0.9]
t = np.linspace(0,30,1000)

sol = odeint(logi, x0, t, args=(mux, muy,mxx,mxy,myx,myy),mxstep=50000)
plt.plot(t, sol[:, 0], 'g', label='x(t)')
plt.plot(t, sol[:, 1], 'r', label='y(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.title('(Mxy,Myx) = (1.5,1.5),(x0,y0) = (.1,.9)')
plt.grid()
plt.show()
