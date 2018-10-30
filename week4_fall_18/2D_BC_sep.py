import module_M_BC as mmBC
import numpy as np
import matplotlib.pyplot as plt
nu,L = mmBC.get_matrix()
print(L)
Delta_M1 = np.array([[0,-0.7],
                     [0,0]])
fig,ax = plt.subplots(figsize=(6,6))
mmBC.parameter_deltaM(Delta_M1,4,nu,L)
fig,ax = plt.subplots(figsize=(6,6))
mmBC.phase_deltaM(Delta_M1,4,nu,L)
plt.show()
