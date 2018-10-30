import numpy as np
import barebones_CDI as bb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import module_M_change as mm

labels, mu, M, eps = bb.get_stein_params()
# stein_ss = dictionary of steady states A-E. We choose to focus on two steady
# states (C and E) 
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C']
    
Delta_M1 = np.array([[0,0.79],
                     [0,0]])
coeff_alpha = np.zeros((11,11))
for i in range(11):
    for j in range(11):
        coeff_alpha[i][j] = ssa[i]*ssb[j]/sum(ssa)
                
Delta_M_4_2 = Delta_M1[0][1]/coeff_alpha[4][2]
print(Delta_M_4_2)


fig, ax = plt.subplots(figsize=(6,6))
num = 30
phase_dict = mm.get_ss_fates(num,0,([4,2]))
sep1 = mm.get_seperatrix_curve(phase_dict,num)
plt.plot(sep1[:,0],sep1[:,1],'--b')
steps = 4
color = ['r','r','r','r']
for i in range(steps):
    phase_dict = mm.get_ss_fates(num,Delta_M_4_2/steps*(i+1),([4,2]))
    sep1 = mm.get_seperatrix_curve(phase_dict,num)
    plt.plot(sep1[:,0],sep1[:,1],color[i],ls='--')


plt.xlabel("SSE",fontsize = 20)
plt.ylabel("SSC",fontsize = 20)

plt.show()
plt.show()