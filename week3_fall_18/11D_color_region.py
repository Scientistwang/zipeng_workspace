import numpy as np
import barebones_CDI as bb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import pickle

def time_change_M(t):
    labels, mu, M, eps = bb.get_stein_params()
    deltaM_4_2 = 0.18520005075054308
    t_change = 80
    coord = ([4,2])
    K = np.zeros((11,11))
    for i in range(11):
        for j in range(11):
            if coord == ([i,j]):
                K[i][j] = M[i][j]+deltaM_4_2
            else:
                K[i][j] = M[i][j]
    if (t<t_change):
        return K
    else:
        return M

def time_change_integrand(x,t,mu,time_change_M):
    dxdt = ( np.dot(np.diag(mu), x)
             + np.dot(np.diag(x), np.dot(time_change_M(t), x)) )
    for i in range(len(x)):
        if abs(x[i]) < 1e-8:
            dxdt[i] = 0
    return dxdt



num_points = 5

labels, mu, M, eps = bb.get_stein_params()
stein_ss = bb.get_all_ss()
ssa = stein_ss['E']; ssb = stein_ss['C']
ssA= stein_ss['A']; ssB = stein_ss['B']
ssD = stein_ss['D']

Delta_M1 = np.array([[0,0.79],
                     [0,0]])
coeff_alpha = np.zeros((11,11))
for i in range(11):
    for j in range(11):
        coeff_alpha[i][j] = ssa[i]*ssb[j]/sum(ssa)
                
Delta_M_4_2 = Delta_M1[0][1]/coeff_alpha[4][2]
print(Delta_M_4_2)
delta_K = Delta_M_4_2
coord = ([4,2])

K = np.zeros((11,11))
for i in range(11):
    for j in range(11):
        if coord == ([i,j]):
            K[i][j] = M[i][j]+delta_K
        else:
            K[i][j] = M[i][j]

phase_dict = {}
max_x = 1
max_y = 1
xs = np.linspace(0, max_x*1.1, num_points)
ys = np.linspace(0, max_y*1.1, num_points)
filename = 'gLV_phases_N_{}'.format(num_points)
# if read_data = True, read the phase values from the disk
# if read_data = False, rerun the simulation and save values to disk
# (run it as False first, then switch to True)
read_data = False
if not read_data:
    for x in xs:
        print(x)
        for y in ys:
            ic = x*ssa + y*ssb
            t = np.linspace(0, 10000, 5001)
            z = integrate.odeint(time_change_integrand, ic, t, args=(mu, time_change_M))
            eps = .01
            if np.linalg.norm(z[-1] - ssa) < eps: color = 'purple'
            elif np.linalg.norm(z[-1] - ssb) < eps: color = 'g'
            
            elif np.linalg.norm(z[-1] - ssA) < eps: color = 'k'
            elif np.linalg.norm(z[-1] - ssB) < eps: color = 'b'
            elif np.linalg.norm(z[-1] - ssD) < eps: color = 'r'
            else: 
                print('{}, {}: neither SS'.format(x, y)); color = 'orange'
                print(z[-1])
                traj_2D = bb.project_to_2D(z, ssa, ssb)
                plt.plot(traj_2D[:,0], traj_2D[:,1],'b',label = 'Old Trajectory')
            phase_dict[(x, y)] = (z[-1], color)
    with open('data/{}'.format(filename), 'wb') as f:
        pickle.dump(phase_dict, f)
        print('... SAVED data to {}'.format(filename))
else:
    with open('data/{}'.format(filename), 'rb') as f:
        phase_dict = pickle.load(f)
        print('... LOADED data from {}'.format(filename))
purples = []
greens = []
blacks = []
blues = []
reds = []
for key in phase_dict:
    if phase_dict[key][1] == 'g':
        greens.append(list(key))
    elif phase_dict[key][1] == 'purple':
        purples.append(list(key))
    elif phase_dict[key][1] == 'k':
        blacks.append(list(key))
    elif phase_dict[key][1] == 'b':
        blues.append(list(key))
    elif phase_dict[key][1] == 'r':
        reds.append(list(key))
    else:
        print('{} didn\'t go to either steady state'.format(key))

greens = np.array(greens)
purples = np.array(purples)
blacks = np.array(blacks)
reds = np.array(reds)
blues = np.array(blues)


markertype = 's'
green = 'green'
purple = 'purple'
black = 'k'
blue = 'b'
red = 'r'
zorder = 0
markersize = 5.3
alpha = 1

#ig, ax = plt.subplots(figsize=(6,6))

plt.plot(greens[:,0], greens[:,1], markertype, color=green,
            zorder=zorder, markersize=markersize, alpha=alpha)
plt.plot(purples[:,0], purples[:,1], markertype, color=purple, zorder=zorder,
          markersize=markersize, alpha=alpha)

plt.xlim(0, max_x*1.1)
plt.ylim(0, max_y*1.1)
#strange destination:[ 0.       0.       0.       0.05698  1.19872  1.12256  0.       0.   1.17829  0.       0.     ]
plt.show()