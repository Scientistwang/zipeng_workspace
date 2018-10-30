#!/usr/bin/env python3

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


###############################################################################


## MAIN FUNCTION


# labels = list of 11 bacteria names; mu = bacterial growth rates (11D);
# M = bacterial interactions (11D);
# eps = susceptibility to antibiotics (we won't use this)
labels, mu, M, eps = bb.get_stein_params()
# stein_ss = dictionary of steady states A-E. We choose to focus on two steady
# states (C and E) 
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C']
# we get the reduced 2D growth rates nu and 2D interactions L through steady
# state reduction
nu, L = bb.SSR(ssa, ssb, mu, M)
# solve the gLV equations for ic=[.5, .5]
ic = [.5, .5]
t = np.linspace(0, 10, 1001)
traj_2D = integrate.odeint(integrand, ic, t, args=(nu, L))

# generate Taylor expansion of separatrix
p = bb.Params((L, [0, 0], nu))
# now p is a Class, and it contains elements p.M, and p.mu, as well as various
# helper functions (e.g. the get_11_ss function, which returns the semistable
# coexistent fixed point
u, v = p.get_11_ss()
print(u, v)
# return Taylor coefficients to 5th order
taylor_coeffs = p.get_taylor_coeffs(order=5)
# create separatrix
xs = np.linspace(0, 1.2, 1001)
ys = np.array([sum([(taylor_coeffs[i]/math.factorial(i))*(x - u)**i for i in range(len(taylor_coeffs))])
               for x in xs])

plt.plot(traj_2D[:,0], traj_2D[:,1],label = 'Trajectory')
plt.plot(xs, ys, color='grey', ls='--',label = 'Separatrix')
plt.axis([0, 1.2, 0, 1.2])
plt.xlabel(r'$x_a$',fontsize = 20)
plt.ylabel(r'$x_b$',fontsize = 20)
plt.tight_layout()
filename = 'figs/example_trajectory.pdf'
plt.scatter([1],[0],facecolor = 'g',edgecolor = 'g',s=80)
plt.scatter([0],[1],facecolor = 'g',edgecolor = 'g',s=80)
plt.scatter([.5],[.5],color = 'k',s=80)
plt.savefig(filename)
plt.legend(prop={'size': 20})
plt.show()