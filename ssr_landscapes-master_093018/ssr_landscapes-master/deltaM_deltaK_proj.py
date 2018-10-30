#!/bin/python

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
import numpy as np
import barebones_CDI as bb

Delta_M1 = np.array([[0,-2.7],
                     [0,0]])  
labels, mu, M, eps = bb.get_stein_params()
# stein_ss = dictionary of steady states A-E. We choose to focus on two steady
# states (C and E) 
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C']
# we get the reduced 2D growth rates nu and 2D interactions L through steady
# state reduction
nu, L = bb.SSR(ssa, ssb, mu, M)
L1 = L - Delta_M1
L = [L0, L1, L2, L3, L4, L5, L6]

coeff = np.zeros((11,11))
for i in range(11):
        for j in range(11):
                coeff[i][j] = ssa[i]*ssb[j]
###############################################
M = []
for LL in L:
    M.append(np.array([LL[0][1]/LL[1][1], L[1][0]/L[0][0]))
# M = [M_0, ...., M_6]
#M_0 = np.array([L[0][1]/(L[1][1]),L[1][0]/L[0][0]])
#M_1 = np.array([L1[0][1]/(L1[1][1]),L1[1][0]/L1[0][0]])
#M_2 = np.array([L2[0][1]/(L1[1][1]),L2[1][0]/L1[0][0]])
#M_3 = np.array([L3[0][1]/(L1[1][1]),L3[1][0]/L1[0][0]])
#M_4 = np.array([L4[0][1]/(L1[1][1]),L4[1][0]/L1[0][0]])
#M_5 = np.array([L5[0][1]/(L1[1][1]),L5[1][0]/L1[0][0]])
#M_6 = np.array([L6[0][1]/(L1[1][1]),L6[1][0]/L1[0][0]])

mu_b = nu[1]/nu[0]
fig, ax = plt.subplots()

colors = ['b', 'b', 'r', 'b', 'b', 'b']

for i,my_M in enumerate(M):
    plt.scatter(my_M[0],my_M[1],facecolor = colors[i],edgecolor = colors[i])


#plt.scatter(M_0[0],M_0[1],facecolor = 'b',edgecolor = 'b')
#plt.scatter(M_1[0],M_1[1],facecolor = 'r',edgecolor = 'r',s = 100)

#plt.scatter(M_2[0],M_2[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_3[0],M_3[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_4[0],M_4[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_5[0],M_5[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_6[0],M_6[1],facecolor = 'r',edgecolor = 'r')

###############################################
Delta_M1 = np.array([[0,-2.7],
                     [0,0]])
Delta_M_4_2 = Delta_M1[0][1]/coeff[4][2] #delta M_ab = coeff[i][j]* Delta M[i][j]
K_new = M
K_new[4][2] = M[4][2]- Delta_M_4_2 
nu_new, L_new = bb.SSR(ssa, ssb, mu, K_new)
M_new = np.array([L_new[0][1]/(L_new[1][1]),L_new[1][0]/L_new[0][0]])
plt.scatter(M_new[0],M_new[1],facecolor = 'g',edgecolor = 'g')

plt.axvline(x=1/mu_b)
plt.axhline(y=mu_b)
plt.axis([-0.15,0.6,0,11])
plt.xlabel(r'$M_{ab}$', fontsize=20)
plt.ylabel(r'$M_{ba}$', fontsize=20)
plt.text(0.16, 11.5, r' $\frac{1}{\mu_b}$', fontsize=20)
plt.text(0.6, 5, r' $\mu_b$', fontsize=20)
plt.xticks([])
plt.yticks([])
filename = 'figs/parameter_space.pdf'
plt.savefig(filename)
plt.show()
