#!/usr/bin/env python3

import numpy as np
import barebones_CDI as bb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import module_M_change as mm

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
###########################################
coeff_alpha = np.zeros((11,11))
for i in range(11):
        for j in range(11):
                coeff_alpha[i][j] = ssa[i]*ssb[j]
#convert coeff matrix into single list
alpha = np.zeros(121)
count = 0
for i in range(11):
        for j in range(11):
            alpha[count] = (coeff_alpha[i][j])
            count +=1
###########################################
coeff_beta = np.zeros((11,11))
for i in range(11):
        for j in range(11):
                coeff_beta[i][j] = ssb[i]*ssa[j]
#convert coeff matrix into single list
beta = np.zeros(121)
count = 0
for i in range(11):
        for j in range(11):
            beta[count] = (coeff_beta[i][j])
            count +=1
###########################################"dot product"
dot = np.dot(alpha,beta)
print(dot)

############################################gamma
gamma = np.zeros(121)
for i in range(121):
    gamma[i] = alpha[i]*beta[i]

mm.histo(25,gamma)
plt.xlabel(r'value of coefficients $\gamma_{ij}$',fontsize = 20)
plt.ylabel('number of coefficients',fontsize = 20)
filename = 'figs/parameter_check.pdf'
plt.savefig(filename)
plt.show()





