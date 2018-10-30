import numpy as np
import barebones_CDI as bb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import pickle

def time_change_M(t):
    labels, mu, M, eps = bb.get_stein_params()
    deltaM_4_2 = 0.18520005075054308
    t_change = 2000
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
    
def integrand(x, t, mu, M):
    """ Return N-dimensional gLV equations """
    dxdt = ( np.dot(np.diag(mu), x)
             + np.dot(np.diag(x), np.dot(M, x)) )
    for i in range(len(x)):
        if abs(x[i]) < 1e-8:
            dxdt[i] = 0
    return dxdt
    
def get_matrix():   ##############gives nu and L from two steady states
    # labels = list of 11 bacteria names; mu = bacterial growth rates (11D);
    # M = bacterial interactions (11D);
    # eps = susceptibility to antibiotics (we won't use this)
    labels, mu, M, eps = bb.get_stein_params()
    # stein_ss = dictionary of steady states A-E. We choose to focus on two steady
    # states (C and E) 
    stein_ss = bb.get_all_ss()
    ssa = stein_ss['E']; ssb = stein_ss['C']
    # we get the reduced 2D growth rates nu and 2D interactions L through steady
    # state reduction
    nu, L = bb.SSR(ssa, ssb, mu, M)
    return nu,L
    
def parameter_deltaM(delta_M,steps,nu,L):  ############## create a plot showing the change in M in steps
    L1 = L + delta_M
    mu_b = nu[1]/nu[0]
    ###############################################
    M_0 = np.array([L[0][1]/(L[1][1]),L[1][0]/L[0][0]])
    M_1 = np.array([L1[0][1]/(L1[1][1]),L1[1][0]/L1[0][0]])
    middle = np.zeros((steps,2))
    for i in range(steps):
        middle[i] = (M_1-M_0)/steps*(i+1)+M_0
        plt.scatter(middle[i][0],middle[i][1],facecolor = 'r',edgecolor = 'r')
    plt.scatter(M_0[0],M_0[1],facecolor = 'b',edgecolor = 'b')
    ###############################################

    plt.axvline(x=1/mu_b)
    plt.axhline(y=mu_b)
    plt.axis([-1,4,-0.5,2])
    plt.xlabel(r'$M_{ab}$', fontsize=20)
    plt.ylabel(r'$M_{ba}$', fontsize=20)
    plt.text(0.16, 11.5, r' $\frac{1}{\mu_b}$', fontsize=20)
    plt.text(0.6, 5, r' $\mu_b$', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    filename = 'figs/parameter_space.pdf'
    plt.savefig(filename)
    #plt.show()
    
def phase_deltaM(delta_M,steps,nu,L):   ############# create a graph showing trajectory and seperatrix in 2D in steps
    # solve the gLV equations for ic=[.5, .5]
    ic = [.5, .5]
    t = np.linspace(0, 500, 5001)
    traj_2D = integrate.odeint(integrand, ic, t, args=(nu, L))
    plt.plot(traj_2D[:,0], traj_2D[:,1],'b',label = 'Old Trajectory')
    # generate Taylor expansion of separatrix
    p = bb.Params((L, [0, 0], nu))
    u, v = p.get_11_ss()
    
    taylor_coeffs = p.get_taylor_coeffs(order=5)
    # create separatrix
    xs = np.linspace(0, 1.2, 1001)
    ys0 = np.array([sum([(taylor_coeffs[i]/math.factorial(i))*(x - u)**i for i in range(len(taylor_coeffs))])
               for x in xs])
    plt.plot(xs, ys0, color='b', ls='--',label = 'Old Separatrix')
    for i in range(steps):
        L_new = delta_M/steps*(i+1)+L
        p_new = bb.Params((L_new, [0, 0], nu))
        u_new, v_new = p_new.get_11_ss()
        taylor_coeffs_new = p_new.get_taylor_coeffs(order=5)
        ys_new = np.array([sum([(taylor_coeffs_new[i]/math.factorial(i))*(x - u_new)**i for i in range(len(taylor_coeffs_new))])
               for x in xs])
        plt.plot(xs, ys_new, color='red', ls='--')
        ##################traj#######################
        traj_2D_new = integrate.odeint(integrand, ic, t, args=(nu, L_new))
        #plt.plot(traj_2D_new[:,0], traj_2D_new[:,1],'r')
        
        
    plt.plot(xs, ys_new, color='red', ls='--', label = 'New Separatrix') #####adding the legend
    plt.plot(traj_2D_new[:,0], traj_2D_new[:,1],'r',label = 'New Trajectory') #####adding the legend
    plt.axis([0, 1.2, 0, 1.2])
    plt.xlabel(r'$x_a$',fontsize = 20)
    plt.ylabel(r'$x_b$',fontsize = 20)
    plt.scatter([1,0,.5],[0,1,.5],facecolor = 'k',edgecolor = 'k')
    plt.tight_layout()
    plt.legend(prop={'size': 20},loc = 1)
    filename = 'figs/phase_space.pdf'
    plt.savefig(filename)

    #plt.show()
    
def convert_to_coeff(ssa,ssb):   ############convert 11D coeffs into a single list
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
    return alpha

def histo(num_bins,coeff_list):   ############ create a graph showing value of coeffs###probably will just use the 1-line code below
    n, bins, patches = plt.hist(coeff_list, num_bins, facecolor='blue', alpha=0.5)
    #plt.show()
    
def parameter_check(alpha, beta):  ######### dot product of two coeffs
    gamma = np.zeros(121)
    for i in range(121):
        gamma[i] = alpha[i]*beta[i]
    return gamma

def get_ss_fates(num_points,delta_K,coord):
    labels, mu, M, eps = bb.get_stein_params()
    stein_ss = bb.get_all_ss()
    ssa = stein_ss['E']; ssb = stein_ss['C']
    ssA = stein_ss['A']; ssB = stein_ss['B']
    ssD = stein_ss['D']
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
                t = np.linspace(0, 1000, 501)
                z = integrate.odeint(integrand, ic, t, args=(mu, K))

                eps = .001
                if np.linalg.norm(z[-1] - ssa) < eps: color = 'purple'
                elif np.linalg.norm(z[-1] - ssb) < eps: color = 'g'
                
                elif np.linalg.norm(z[-1] - ssA) < eps: color = 'k'
                elif np.linalg.norm(z[-1] - ssB) < eps: color = 'b'
                elif np.linalg.norm(z[-1] - ssD) < eps: color = 'r'
                else: print('{}, {}: neither SS'.format(x, y)); color = 'orange'
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

 #   fig, ax = plt.subplots(figsize=(6,6))

 #   ax.plot(greens[:,0], greens[:,1], markertype, color=green,
 #           zorder=zorder, markersize=markersize, alpha=alpha)
 #   ax.plot(purples[:,0], purples[:,1], markertype, color=purple, zorder=zorder,
#          markersize=markersize, alpha=alpha)
 #   if blacks != []:
 #       ax.plot(blacks[:,0], blacks[:,1], markertype, color=black, zorder=zorder,
 #           markersize=markersize, alpha=alpha) 
 #   if blues != []:       
 #       ax.plot(blues[:,0], blues[:,1], markertype, color=blue, zorder=zorder,
 #           markersize=markersize, alpha=alpha) 
  #  if reds != []:       
  #      ax.plot(reds[:,0], reds[:,1], markertype, color=red, zorder=zorder,
  #          markersize=markersize, alpha=alpha)   

  #  ax.set_xlim(0, max_x*1.1)
   # ax.set_ylim(0, max_y*1.1)

    savefig = False
    if savefig:
        plt.savefig('figs/{}_plot.pdf'.format(filename), bbox_inches='tight')
        print('... saved to {}_plot.pdf'.format(filename))
        
    return phase_dict;
    
def get_seperatrix_curve(phase_dict,num_points):
    max_x = 1
    max_y = 1
    xs = np.linspace(0, max_x*1.1, num_points)
    ys = np.linspace(0, max_y*1.1, num_points)
    pt = []
    for i in range(0,len(xs),1):
        for j in range(0,len(ys)-1,1):
            if phase_dict[(xs[i],ys[j])][1] == 'purple':
                if phase_dict[(xs[i],ys[j+1])][1] == 'g':
                    pt.append([xs[i],(ys[j]+ys[j+1])/2])
                elif phase_dict[(xs[i],ys[j+1])][1] == 'orange':
                    pt.append([xs[i],(ys[j]+ys[j+1])/2])
            elif phase_dict[(xs[i],ys[j])][1] == 'orange':
                if phase_dict[(xs[i],ys[j+1])][1] == 'g':
                    pt.append([xs[i],(ys[j]+ys[j+1])/2])
    for i in range(0,len(ys),1):
        for j in range(0,len(xs)-1,1):
            if phase_dict[(xs[j],ys[i])][1] == 'purple':
                if phase_dict[(xs[j+1],ys[i])][1] == 'g':
                    pt.append([(xs[j]+xs[j+1])/2,ys[i]])
    
    pt = np.array(pt)
    return pt;
    
def _11D_traj(delta_K, coord):
    t = np.linspace(0, 1000, 1001)
    stein_ss = bb.get_all_ss()
    ssa = stein_ss['E']; ssb = stein_ss['C']
    ic = 0.5*ssa+ssb*0.5
    labels, mu, M, eps = bb.get_stein_params()
    K = np.zeros((11,11))
    for i in range(11):
        for j in range(11):
            if coord == ([i,j]):
                K[i][j] = M[i][j]+delta_K
            else:
                K[i][j] = M[i][j]
    traj = integrate.odeint(integrand, ic, t, args=(mu, K))
    print(traj[-1])
    traj_2D = bb.project_to_2D(traj, ssa, ssb)
    return traj_2D
    
def get_11D_traj():   ########uses _11D_traj to create plots
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
    traj_2D = _11D_traj(Delta_M_4_2,([4,2]))
    plt.plot(traj_2D[:,0], traj_2D[:,1],'b',label = 'Old Trajectory')
    traj_2D_old = _11D_traj(0,([4,2]))
    plt.plot(traj_2D_old[:,0], traj_2D_old[:,1],'r',label = 'Old Trajectory')
    
    
def time_change_11D_traj(delta_K, coord):
    t = np.linspace(0, 10000, 10001)
    stein_ss = bb.get_all_ss()
    ssa = stein_ss['E']; ssb = stein_ss['C']
    ic = 0.33*ssa+ssb*0.55
    labels, mu, M, eps = bb.get_stein_params()
    K = np.zeros((11,11))
    for i in range(11):
        for j in range(11):
            if coord == ([i,j]):
                K[i][j] = M[i][j]+delta_K
            else:
                K[i][j] = M[i][j]
    traj = integrate.odeint(time_change_integrand, ic, t, args=(mu, time_change_M))
    print(traj[-1])
    traj_2D = bb.project_to_2D(traj, ssa, ssb)
    return traj_2D
################################main
if __name__ == "__main__":
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
    traj_2D = time_change_11D_traj(Delta_M_4_2,([4,2]))
    plt.plot(traj_2D[:,0], traj_2D[:,1],'b',label = 'Old Trajectory')
    traj_2D_old = _11D_traj(0,([4,2]))
    plt.plot(traj_2D_old[:,0], traj_2D_old[:,1],'r',label = 'Old Trajectory')
    plt.show()
    
    
    
    
