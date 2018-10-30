import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib
from scipy import integrate
from scipy.integrate import odeint

#x'(t) = x(t)*(mu-M*x(t))
def logi(x,t,mu,m):
    x1 = x
    dxdt = x*(mu-m*x)
    return dxdt

mu = 1
m = 0.5
x0 = 1
t = np.linspace(0,10,101)

sol = odeint(logi, x0, t, args=(mu, m))
plt.plot(t, sol[:, 0], 'g', label='x(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()