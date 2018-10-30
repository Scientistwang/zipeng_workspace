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


labels, mu, M, eps = bb.get_stein_params()
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C']

coeff = np.zeros((11,11))
for i in range(11):
        for j in range(11):
                coeff[i][j] = ssa[i]*ssb[j]
##########################################convert coeff matrix into single list
coeff_list = np.zeros(121)
count = 0
for i in range(11):
        for j in range(11):
            coeff_list[count] = (coeff[i][j])
            count +=1
##############################################plot the histogram
plt.figure(0)
num_bins = 25
n, bins, patches = plt.hist(coeff_list, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel(r'value of coefficients $\alpha_{ij}$',fontsize = 20)
plt.ylabel('number of coefficients',fontsize = 20)
filename = 'figs/histogram.pdf'
plt.savefig(filename)
#coeff[4][2] = 15.11
###################################### original trajectory
plt.figure(1)
ic = (ssa+ssb)*0.5
t = np.linspace(0, 100, 10001)
traj = integrate.odeint(integrand, ic, t, args=(mu, M))
traj_2D = bb.project_to_2D(traj, ssa, ssb)
plt.plot(traj_2D[:,0], traj_2D[:,1],'b',label = 'Old Trajectory')
############################################change matrix M
Delta_M1 = np.array([[0,-2.7],
                     [0,0]])
Delta_M_4_2 = Delta_M1[0][1]/coeff[4][2] #delta M_ab = coeff[i][j]* Delta M[i][j]
M_new = M
M_new[4][2] = M[4][2]- Delta_M_4_2
traj_new = integrate.odeint(integrand, ic, t, args=(mu, M_new))
traj_2D_new = bb.project_to_2D(traj_new, ssa, ssb)
plt.plot(traj_2D_new[:,0], traj_2D_new[:,1],'r',label = 'New Trajectory')
plt.xlabel('SSE',fontsize = 20)
plt.ylabel('SSC',fontsize = 20)
plt.scatter([1,0],[0,1],facecolor = 'g',edgecolor = 'g',s=20)
plt.scatter([.5],[.5],facecolor = 'k',edgecolor = 'k',s=20)
plt.legend(prop={'size': 20})
plt.show()
filename = 'figs/11D_trajectory.pdf'
plt.savefig(filename)
plt.show()




