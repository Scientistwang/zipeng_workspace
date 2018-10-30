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
change_L = Delta_M1             #change from M1 to M6
L_new = L - change_L
L_middle1 = L-change_L/4
L_middle2 = L-change_L/2
L_middle3 = L-change_L*3/4
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
xs = np.linspace(0, 1.2, 1001)
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
#################################################################
p_middle1 = bb.Params((L_middle1, [0, 0], nu))
u_middle1, v_middle1 = p_middle1.get_11_ss()
taylor_coeffs_middle1 = p_middle1.get_taylor_coeffs(order=5)
ys_middle1 = np.array([sum([(taylor_coeffs_middle1[i]/math.factorial(i))*(x - u_middle1)**i for i in range(len(taylor_coeffs_middle1))])
               for x in xs])
#################################################################
#################################################################
p_middle2 = bb.Params((L_middle2, [0, 0], nu))
u_middle2, v_middle2 = p_middle2.get_11_ss()
taylor_coeffs_middle2 = p_middle2.get_taylor_coeffs(order=5)
ys_middle2 = np.array([sum([(taylor_coeffs_middle2[i]/math.factorial(i))*(x - u_middle2)**i for i in range(len(taylor_coeffs_middle2))])
               for x in xs])
#################################################################
#################################################################
p_middle3 = bb.Params((L_middle3, [0, 0], nu))
u_middle3, v_middle3 = p_middle3.get_11_ss()
taylor_coeffs_middle3 = p_middle3.get_taylor_coeffs(order=5)
ys_middle3 = np.array([sum([(taylor_coeffs_middle3[i]/math.factorial(i))*(x - u_middle3)**i for i in range(len(taylor_coeffs_middle3))])
               for x in xs])
#################################################################
plt.plot(traj_2D[:,0], traj_2D[:,1],'b',label = 'Old Trajectory')
plt.plot(traj_2D_new[:,0], traj_2D_new[:,1],'r',label = 'New Trajectory')
plt.plot(xs, ys, color='b', ls='--',label = 'Old Separatrix')
plt.plot(xs, ys_new, color='red', ls='--',label = 'New Separatrix')
plt.plot(xs, ys_middle1, color='red', ls='--',label = 'Middle Separatrix1')
plt.plot(xs, ys_middle2, color='red', ls='--',label = 'Middle Separatrix2')
plt.plot(xs, ys_middle3, color='red', ls='--',label = 'Middle Separatrix3')
plt.axis([0, 1.2, 0, 1.2])
plt.xlabel(r'$x_a$',fontsize = 20)
plt.ylabel(r'$x_b$',fontsize = 20)
plt.scatter([1,0],[0,1],facecolor = 'k',edgecolor = 'k')
plt.tight_layout()
filename = 'figs/separatrix_deltaM6.pdf'
plt.savefig(filename)
#plt.legend(prop={'size': 20})
plt.show()