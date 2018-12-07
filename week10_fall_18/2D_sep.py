import module_M_change as mm
import numpy as np
import matplotlib.pyplot as plt
nu,L = mm.get_matrix()
Delta_M1 = np.array([[0,0.79],
                     [0,0]])
fig,ax = plt.subplots(figsize=(6,6))
mm.parameter_deltaM(Delta_M1,4,nu,L)
fig,ax = plt.subplots(figsize=(6,6))
mm.phase_deltaM(Delta_M1,4,nu,L)
plt.show()