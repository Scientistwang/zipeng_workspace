import module_M_change as mmBC
import barebones_CDI as bb
import numpy as np
import matplotlib.pyplot as plt
nu,L = mmBC.get_matrix()
Delta_M1 = np.array([[0,0.79],
                     [0, 0  ]])
labels, mu, M, eps = bb.get_stein_params()
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C' ]
coeff_matrix = np.zeros((11,11))
coeff_list = np.zeros(121)
count = 0
for i in range(11):
    for j in range(11):
        coeff_list[count] = ssa[i]*ssb[j]/sum(ssa)
        coeff_matrix[i][j] = ssa[i]*ssb[j]/sum(ssa)
        count+=1
mmBC.histo(20,coeff_list)
print(max(coeff_list))
print(coeff_matrix)
#coord = (2,5)
print(coeff_matrix[2][5])
plt.show()




