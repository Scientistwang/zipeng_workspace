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

for delta_type, M_name in zip([Delta_M1, Delta_M2, ...], ['M1', 'M2', ...]):
    print(delta_type, M_name)



###############################################################################
Delta_M1 = np.array([[0,-2.7],
                     [0,0]])
Delta_M2 = np.array([[0,-2.5],
                     [1,0]])
Delta_M3 = np.array([[0,-2.2],
                     [2,0]])
Delta_M4 = np.array([[0,-0],
                     [7.4,0]])
Delta_M5 = np.array([[0,-1],
                     [5.1,0]])                     
Delta_M6 = np.array([[0,-2],
                     [2.4,0]])                     
                     
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
#################################################################
change_L = Delta_M6             #change from M1 to M6
L_new = L - change_L
print(L)
###################################################################
# solve the gLV equations for ic=[.5, .5]
ic = [.5, .5]
t = np.linspace(0, 20, 1001)
traj_2D = integrate.odeint(integrand, ic, t, args=(nu, L))
traj_2D_new = integrate.odeint(integrand, ic, t, args=(nu, L_new))#added
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
xs = np.linspace(0, 1, 1001)
ys = np.array([sum([(taylor_coeffs[i]/math.factorial(i))*(x - u)**i for i in range(len(taylor_coeffs))])
               for x in xs])
#################################################################
p_new = bb.Params((L_new, [0, 0], nu))
u_new, v_new = p_new.get_11_ss()
print(u_new, v_new)
taylor_coeffs_new = p_new.get_taylor_coeffs(order=5)
ys_new = np.array([sum([(taylor_coeffs_new[i]/math.factorial(i))*(x - u_new)**i for i in range(len(taylor_coeffs_new))])
               for x in xs])
#################################################################
plt.figure()
plt.plot(traj_2D[:,0], traj_2D[:,1],'b')
plt.plot(traj_2D_new[:,0], traj_2D_new[:,1],'r')
plt.plot(xs, ys, color='grey', ls='--')
plt.plot(xs, ys_new, color='red', ls='--')
plt.axis([0, 1, 0, 1])
plt.tight_layout()
filename = 'figs/separatrix_deltaM6.pdf'
filename = 'figs/separatrix_delta{}.pdf'.format(M_name)

plt.savefig(filename)
plt.show()
