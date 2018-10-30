#!/usr/bin/env python3

import numpy as np
import barebones_CDI as bb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import pickle


def integrand(x, t, mu, M):
    """ Return N-dimensional gLV equations """
    dxdt = ( np.dot(np.diag(mu), x)
             + np.dot(np.diag(x), np.dot(M, x)) )
    for i in range(len(x)):
        if abs(x[i]) < 1e-8:
            dxdt[i] = 0
    return dxdt

def how_to_get_separatrix():
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
    # solve the gLV equations for ic=[.5, .5]
    ic = [.5, .5]
    t = np.linspace(0, 10, 1001)
    traj_2D = integrate.odeint(integrand, ic, t, args=(nu, L))

    # generate Taylor expansion of separatrix
    p = bb.Params((L, [0, 0], nu))
    # now p is a Class, and it contains elements p.M, and p.mu, as well as various
    # helper functions (e.g. the get_11_ss function, which returns the semistable
    # coexistent fixed point
    u, v = p.get_11_ss()
    # return Taylor coefficients to 5th order
    taylor_coeffs = p.get_taylor_coeffs(order=5)
    # create separatrix
    xs = np.linspace(0, 1, 1001)
    ys = np.array([sum([(taylor_coeffs[i]/math.factorial(i))*(x - u)**i for i in range(len(taylor_coeffs))])
                   for x in xs])

    plt.plot(traj_2D[:,0], traj_2D[:,1])
    plt.plot(xs, ys, color='grey', ls='--')
    plt.axis([0, 1, 0, 1])
    plt.tight_layout()
    filename = 'figs/example_trajectory.pdf'
    plt.savefig(filename)

def get_ss_fates():
    labels, mu, M, eps = bb.get_stein_params()
    stein_ss = bb.get_all_ss()
    ssa = stein_ss['E']; ssb = stein_ss['C']

    phase_dict = {}
    max_x = 1
    max_y = 1
    num_points = 10 # 11 is saved
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
                z = integrate.odeint(integrand, ic, t, args=(mu, M))

                eps = .001
                if np.linalg.norm(z[-1] - ssa) < eps: color = 'purple'
                elif np.linalg.norm(z[-1] - ssb) < eps: color = 'g'
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
    for key in phase_dict:
        if phase_dict[key][1] == 'g':
            greens.append(list(key))
        elif phase_dict[key][1] == 'purple':
            purples.append(list(key))
        else:
            print('{} didn\'t go to either steady state'.format(key))

    greens = np.array(greens)
    purples = np.array(purples)

    markertype = 's'
    green = 'green'
    purple = 'purple'
    zorder = 0
    markersize = 5.3
    alpha = 1

    fig, ax = plt.subplots(figsize=(6,6))

    ax.plot(greens[:,0], greens[:,1], markertype, color=green,
            zorder=zorder, markersize=markersize, alpha=alpha)
    ax.plot(purples[:,0], purples[:,1], markertype, color=purple, zorder=zorder,
            markersize=markersize, alpha=alpha)

    ax.set_xlim(0, max_x*1.1)
    ax.set_ylim(0, max_y*1.1)

    savefig = True
    if savefig:
        plt.savefig('figs/{}_plot.pdf'.format(filename), bbox_inches='tight')
        print('... saved to {}_plot.pdf'.format(filename))

###############################################################################


## MAIN FUNCTION

get_ss_fates()

