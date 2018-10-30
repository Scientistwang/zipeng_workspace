import numpy as np
import barebones_CDI as bb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import module_M_BC as mmBC

labels, mu, M, eps = bb.get_stein_params()
# stein_ss = dictionary of steady states A-E. We choose to focus on two steady
# states (C and E) 
stein_ss = bb.get_all_ss()
ssa = stein_ss['C']; ssb = stein_ss['B']
    
Delta_M1 = np.array([[0,-0.7],
                             [0,0]])
coeff_alpha = np.zeros((11,11))
for i in range(11):
        for j in range(11):
                coeff_alpha[i][j] = ssa[i]*ssb[j]/sum(ssa)
                                        
Delta_M_2_5 = Delta_M1[0][1]/coeff_alpha[2][5]
print(Delta_M_2_5)
fig, ax = plt.subplots(figsize=(6,6))
traj_2D_old =mmBC._11D_traj(0,([2,5]))
plt.plot(traj_2D_old[:,0], traj_2D_old[:,1],'r',label = 'Old Trajectory')
traj_2D_new =mmBC._11D_traj(Delta_M_2_5,([2,5]))
plt.plot(traj_2D_new[:,0], traj_2D_new[:,1],'b',label = 'New Trajectory')

num = 30
phase_dict = mmBC.get_ss_fates(num,0,([2,5]))
sep1 = mmBC.get_seperatrix_curve(phase_dict,num)
plt.plot(sep1[:,0],sep1[:,1],'--b')
steps = 4
color = ['r','b','g','k']
for i in range(steps):
        phase_dict = mmBC.get_ss_fates(num,Delta_M_2_5/steps*(i+1),([2,5]))
        sep1 = mmBC.get_seperatrix_curve(phase_dict,num)
        plt.plot(sep1[:,0],sep1[:,1],color[i],ls='--')


plt.xlabel("SSC",fontsize = 20)
plt.ylabel("SSB",fontsize = 20)

plt.show()
