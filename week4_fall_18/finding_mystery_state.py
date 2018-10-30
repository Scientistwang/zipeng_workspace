import numpy as np
import barebones_CDI as bb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math

def integrand(x, t, mu, M):
    """ Return N-dimensional gLV equations """
    dxdt = ( np.dot(np.diag(mu), x)
             + np.dot(np.diag(x), np.dot(M, x)) )
    for i in range(len(x)):
        if abs(x[i]) < 1e-8:
            dxdt[i] = 0
    return dxdt

def time_change_M(t):
    labels, mu, M, eps = bb.get_stein_params()
    deltaM_4_2 = 0.18520005075054308
    t_change = 2000
    coord = ([4,2])
    K = np.zeros((11,11))
    for i in range(11):
        for j in range(11):
            if coord == ([i,j]):
                K[i][j] = M[i][j]+deltaM_4_2
            else:
                K[i][j] = M[i][j]
    if (t<t_change):
        return K
    else:
        return M

def time_change_integrand(x,t,mu,time_change_M):
    dxdt = ( np.dot(np.diag(mu), x)
             + np.dot(np.diag(x), np.dot(time_change_M(t), x)) )
    for i in range(len(x)):
        if abs(x[i]) < 1e-8:
            dxdt[i] = 0
    return dxdt


labels, mu, M, eps = bb.get_stein_params()
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C']
#strange destination:[ 0.       0.       0.       0.05698  1.19872  1.12256  0.       0.   1.17829  0.       0.     ]
#ssa = [ 0.     ,  0.     ,  0.     ,  0.00599,  1.22839,  1.10552, 0.     ,  0.03519,  1.16942,  0.     ,  0.     ]
#two microbes are gone in this new state. Try shorten the time spent in modified parameters
#when the time is above 180, they go to this "mysterious state"
#when the time becomes shorter, the white region gradually disappears, and turns green.
mystery_state = np.array([0,0,0,0.05698,1.19872,1.22256,0,0,1.17829,0,0])
ic = mystery_state
t = np.linspace(0, 50000, 5001)
traj = integrate.odeint(integrand, ic, t, args=(mu, M))
print(traj[-1])
traj_2D = bb.project_to_2D(traj,ssa,ssb)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(traj_2D[:,0],traj_2D[:,1],'b')
ax.scatter([0,1],[1,0])
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
plt.show()


