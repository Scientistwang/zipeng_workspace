
import matplotlib.pyplot as plt
import numpy as np
import barebones_CDI as bb

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
                     
labels, mu, M, eps = bb.get_stein_params()
# stein_ss = dictionary of steady states A-E. We choose to focus on two steady
# states (C and E) 
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C']
# we get the reduced 2D growth rates nu and 2D interactions L through steady
# state reduction
nu, L = bb.SSR(ssa, ssb, mu, M)
L1 = L - Delta_M1
L2 = L - Delta_M2
L3 = L - Delta_M3
L4 = L - Delta_M4
L5 = L - Delta_M5
L6 = L - Delta_M6

###############################################
M_0 = np.array([L[0][1]/(L[1][1]),L[1][0]/L[0][0]])
M_1 = np.array([L1[0][1]/(L1[1][1]),L1[1][0]/L1[0][0]])
#M_2 = np.array([L2[0][1]/(L1[1][1]),L2[1][0]/L1[0][0]])
#M_3 = np.array([L3[0][1]/(L1[1][1]),L3[1][0]/L1[0][0]])
#M_4 = np.array([L4[0][1]/(L1[1][1]),L4[1][0]/L1[0][0]])
#M_5 = np.array([L5[0][1]/(L1[1][1]),L5[1][0]/L1[0][0]])
#M_6 = np.array([L6[0][1]/(L1[1][1]),L6[1][0]/L1[0][0]])

mu_b = nu[1]/nu[0]
fig, ax = plt.subplots()

plt.scatter(M_0[0],M_0[1],facecolor = 'b',edgecolor = 'b')
plt.scatter(M_1[0],M_1[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_2[0],M_2[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_3[0],M_3[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_4[0],M_4[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_5[0],M_5[1],facecolor = 'r',edgecolor = 'r')
#plt.scatter(M_6[0],M_6[1],facecolor = 'r',edgecolor = 'r')

###############################################

plt.axvline(x=1/mu_b)
plt.axhline(y=mu_b)
plt.axis([-0.15,0.6,0,11])
plt.xlabel(r'$M_{ab}$', fontsize=20)
plt.ylabel(r'$M_{ba}$', fontsize=20)
plt.text(0.16, 11.5, r' $\frac{1}{\mu_b}$', fontsize=20)
plt.text(0.6, 5, r' $\mu_b$', fontsize=20)
plt.xticks([])
plt.yticks([])
plt.arrow(M_0[0],M_0[1], M_1[0]-M_0[0]+0.03,0, head_width=0.5, head_length=0.02, fc='k', ec='k')
filename = 'figs/parameter_space.pdf'
plt.savefig(filename)
plt.show()