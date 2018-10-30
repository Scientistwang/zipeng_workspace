#!/usr/bin/env python3
#
# Reduction of a gene regulatory network (the bistable switch) into a
# steady-state reduced form, where the two stable states of the gene circuit
# correspond to the steady states (1, 0) and (0, 1) in the reduced
# 2-dimensional generalized Lotka-Volterra system.


import time
t0 = time.time()
import numpy as np
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=4)
import pickle
import matplotlib as mpl

class originalParams:
    """ This class refers to the original 2-dimensional form of the bistable
    switch. By defining all methods that relate to this form in this
    originalParams class, we allow for easy parameter passing and efficient
    searching of parameter space."""
    def __init__(s, params=None):
        if params:
            s.alpha, s.beta, s.gamma = params
        else:
            s.alpha = 4.09
            s.beta = .73
            s.gamma = .2

        sss = s.get_stable_fps()
        s.ssa = sss[0]
        s.ssb = sss[1]
        s.fps = np.array([s.ssa, s.ssb])

        s.ext = 'a{:.3}_b{:.3}_g{:.3}'.format(s.alpha, s.beta, s.gamma)

    def get_integrand(s, Z, t):
        """ Synthetic gene circuit for a bistable switch. Z = [gene x, gene y].
        Genes x and y inhibit each other and themselves Under the parameter
        regimes we are interested in, this differential equation has 2 stable
        steady states. """
        return np.array([s.alpha - s.alpha*Z[1]**2/(1 + Z[1]**2) - Z[0],
                         s.beta - s.beta*Z[0]**2/(1 + Z[0]**2) - s.gamma*Z[1]])
        #return np.array([s.alpha/(1 + Z[1]**2) - Z[0],
        #                 s.beta/(1 + Z[0]**2) - s.gamma*Z[1]])

    def get_jacobian(s, fp):
        """ Return the jacobian of the bistable switch evaluated at the fixed
        point fp"""
        Z = fp
        jac = np.zeros((2,2))
        jac[0][0] = -1
        jac[0][1] = -2*s.alpha*Z[1]/((1 + Z[1]**2)**2)
        jac[1][0] = -2*s.beta*Z[0]/((1 + Z[0]**2)**2)
        jac[1][1] = -s.gamma
        return jac

    def convert_sp_to_op(s, points):
        """ Take a list of 2D points (corresponding to ssrParams) and
        output a list of the corresponding 2D points in the original basis. 
        """
        new_points = []
        for point in points:
            new_point = s.ssa*point[0] + s.ssb*point[1]
            new_points.append(new_point)
        new_points = np.array(new_points)
        return new_points

    def get_nullclines(s):
        """ Returns nullclines of bistable switch dynamical system """
        x_range = np.linspace(0, s.alpha*1.1, 100000)
        y_range = np.linspace(0, s.beta/s.gamma*1.1, 100000)
        x_nullcline = np.array([[s.alpha/(1 + y**2), y] for y in y_range])
        y_nullcline = np.array([[x, (s.beta / s.gamma)/(1 + x**2)] for x in x_range])
        return x_nullcline[x_nullcline[:,0].argsort()], y_nullcline[y_nullcline[:,0].argsort()]

    def get_y_nullcline(s, x_val):
        """ Return the y-value of the y nullcline as a function of x. Since the
        y nullcline is natively a function of x, this is computed exactly. """
        exact_y_val = (s.beta / s.gamma)/(1 + x_val**2)
        return exact_y_val

    def get_x_nullcline(s, x_val, x_null_interp, idx):
        """ Return the y-value of the x nullcline as a function of x. This
        requires adaptive interpolation of the x nullcline, in order to have
        fine resolution"""
        zoomed_y_range = np.linspace(x_null_interp[idx-1],
                x_null_interp[idx+1], 100000)
        zoomed_x_range = np.array([s.alpha/(1 + y**2) for y in zoomed_y_range])
        interpolated_y_val = np.interp(x_val, zoomed_x_range, zoomed_y_range)
        #print('zoomed y range is', min(interp_y_range), max(interp_y_range))
        #print('zoomed x range is', min(interp_x_range), max(interp_x_range))
        #print(x_val, exact_y_val, interpolated_y_val)
        return interpolated_y_val

    def get_nullcline_diff(s, x_val, x_null_interp, idx):
        """ Get difference in nullclines at a particular x value"""
        exact_y_val = s.get_y_nullcline(x_val)
        interpolated_y_val = s.get_x_nullcline(x_val, x_null_interp, idx)
        return exact_y_val - interpolated_y_val

    def plot_nullclines(s, ax, savefig=True):
        """ Plots nullclines of the bistable switch with steady states labeled
        (stable = black, unstable = red) """
        x_nullcline, y_nullcline = s.get_nullclines()
        fps = s.get_intersections(x_nullcline, y_nullcline)
        is_stables = s.get_fp_stability(fps)

        for i,fp in enumerate(fps):
            if i == 0: color='green'
            if i == 1: color='grey'
            if i == 2: color='purple'
            ax.plot(fp[0], fp[1], c=color, marker='.', markeredgecolor='black',
                    markeredgewidth=2, ms=25, zorder=4)

        ax.plot(x_nullcline[:,0], x_nullcline[:,1], c='dimgrey', label='$\dot{x} = 0$')
        ax.plot(y_nullcline[:,0], y_nullcline[:,1], c='dimgrey', label='$\dot{y} = 0$')
        #ax.legend(loc='upper right', framealpha=1)
        ax.set_xlabel('gene $x$ concentration'); ax.set_ylabel('gene $y$ concentration')
        ax.set_title('Nullclines with $\\alpha =${:.3}, $\\beta = ${:.3}, and $\gamma = ${:.3}'
                     .format(s.alpha, s.beta, s.gamma), fontsize=18)
        #ax.axis([0, max(y_nullcline[:,0]), 0, max(x_nullcline[:,1])])
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 1, 2, 3, 4])

        filename = 'figs/op_nullclines_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))

    def plot_example_traj(s, ax, savefig=True):
        ic = np.array([1.8, 3])
        ax.plot(ic[0], ic[1], 'k.', ms=17, zorder=5)
        ax.text(ic[0], ic[1]*1.05, 'IC', ha='center')
        t = np.linspace(0, 500, 501)
        y = integrate.odeint(s.get_integrand, ic, t)
        if False:
            print('orig params trajectory:')
            print(y)
        ax.plot(y[:,0], y[:,1], '--k', zorder=3, label='original form')
        ax.legend()

        filename = 'figs/op_example_traj_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))


    def get_intersections(s, x_null, y_null):
        """ Identify and return when the x and y nullclines intersect"""
        standard_x = np.linspace(0, min(x_null[-1][0], y_null[-1][0]), 100000)
        x_null_interp = np.interp(standard_x, x_null[:,0], x_null[:,1])
        y_null_interp = np.interp(standard_x, y_null[:,0], y_null[:,1])

        interp_diff = x_null_interp - y_null_interp
        # from https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value
        intersection_idx = np.argwhere(np.diff(np.sign(interp_diff)) != 0).reshape(-1)
        intersections = np.array([standard_x[idx] for idx in intersection_idx])
        err = max(np.array([abs(standard_x[idx+1] - standard_x[idx]) for idx in intersection_idx]))
        #print('error is', err)
        true_fps = []
        for idx, inter in zip(intersection_idx, intersections):
            x_star = optimize.brentq(s.get_nullcline_diff, inter-err, inter+err,
                                     args=(x_null_interp, idx))
            y_star = s.get_y_nullcline(x_star)
            true_fps.append([x_star, y_star])
            #print('fixed point is ({:.6}, {:.6})'.format(x_star, y_star))
        true_fps = np.array(true_fps)
        return true_fps

    def get_fp_stability(s, fps):
        """ Evaluate and return whether fixed points fps are stable """
        is_stables = []
        for fp in fps:
            jac = s.get_jacobian(fp)
            evals = np.linalg.eig(jac)[0]
            is_stables.append(all(evals < 0))
        return is_stables

    def get_stable_fps(s):
        """ Return a list of the two stable fixed points of the bistable switch
        system """
        x_nullcline, y_nullcline = s.get_nullclines()
        fps = s.get_intersections(x_nullcline, y_nullcline)
        is_stables = s.get_fp_stability(fps)

        stab_fps = []
        for fp,is_stable in zip(fps, is_stables):
            if is_stable:
                stab_fps.append(fp)
        stab_fps = np.array(stab_fps)[::-1]

        return stab_fps

class ssrParams:
    """ This class contains all methods relevant for the steady state reduction
    of the gLV differential equation form of the bistable switch. """
    def __init__(s, op=originalParams()):

        s.alpha, s.beta, s.gamma = op.alpha, op.beta, op.gamma
        s.op_ssa = op.ssa
        s.op_ssb = op.ssb

        s.ssa = s.convert_op_to_sp([s.op_ssa])[0]
        s.ssb = s.convert_op_to_sp([s.op_ssb])[0]
        s.fps = np.array([s.ssa, s.ssb])

        s.mu, s.M = s.get_ssr_params(op)

        s.ext = 'a{:.3}_b{:.3}_g{:.3}'.format(s.alpha, s.beta, s.gamma)

    def get_integrand(s, Y, t):
        """ Synthetic gene circuit for a bistable switch in SSR reduced 
        form. This integrand returns the derivative of the vector
        [ssa, ssb], where ssa and ssb are the steady states of the gLV form
        equations. """
        val = np.dot(np.diag(s.mu), Y) + np.dot( np.diag(np.dot(s.M, Y)), Y)
        return val

    def get_jacobian(s, fp):
        """ Return the jacobian of the bistable switch in quasipolynomial
        coordinates evaluated at the fixed point fp. """
        N = len(fp)
        x = fp
        jac = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i is j:
                    val = s.mu[i] + np.dot(s.M, x)[i] + s.M[i,i]*x[i]
                    jac[i, j] = val
                else:
                    val = x[i]*s.M[i,j]
                    jac[i, j] = val
        return jac

    def get_fp_stability(s, fps):
        """ Evaluate and return whether fixed points fps are stable """
        is_stables = []
        for fp in fps:
            jac = s.get_jacobian(fp)
            evals = np.linalg.eig(jac)[0]
            is_stables.append(all(evals < 0))
        return is_stables

    def convert_op_to_sp(s, points):
        """ Take a list of 2D points (corresponding to originalParams) and
        output a list of the corresponding 2D points in the SSR basis ([x, y]).
        """
        new_points = []
        for point in points:
            uu = np.dot(s.op_ssa, s.op_ssa); vv = np.dot(s.op_ssb, s.op_ssb)
            xu = np.dot(point, s.op_ssa); xv = np.dot(point, s.op_ssb)
            uv = np.dot(s.op_ssa, s.op_ssb)
            new_points.append([(xu*vv - xv*uv)/(uu*vv - uv**2),
                               (uu*xv - xu*uv)/(uu*vv - uv**2)])
        new_points = np.array(new_points)
        return new_points

    def get_ssr_params(s, op):
        ya,za = op.ssa
        yb,zb = op.ssb
        M_aa = (-ya - 2*op.alpha*za**2/(1 + za**2)**2
                - 2*op.beta*ya**2/(1+ya**2)**2 - op.gamma*za)
        M_ab = (-yb - 2*op.alpha*za*zb/(1 + za**2)**2
                - 2*op.beta*ya*yb/(1+ya**2)**2 - op.gamma*zb)
        M_ba = (-ya - 2*op.alpha*za*zb/(1 + zb**2)**2
                - 2*op.beta*ya*yb/(1+yb**2)**2 - op.gamma*za)
        M_bb = (-yb - 2*op.alpha*zb**2/(1 + zb**2)**2
                - 2*op.beta*yb**2/(1+yb**2)**2 - op.gamma*zb)
        M = np.array([[M_aa, M_ab], [M_ba, M_bb]])
        mu = np.array([-M_aa, -M_bb])
        return mu, M

    def get_mixed_fp(s):
        """ Get coexistent steady state of 2D gLV model """
        xa = (s.M[1][1]*s.mu[0] - s.M[0][1]*s.mu[1])/(s.M[0][1]*s.M[1][0] - s.M[0][0]*s.M[1][1])
        xb = (-s.M[1][0]*s.mu[0] + s.M[0][0]*s.mu[1])/(s.M[0][1]*s.M[1][0] - s.M[0][0]*s.M[1][1])
        return np.array([xa, xb])

    def plot_nullclines(s, ax, savefig=True):
        """ Plots nullclines of the bistable switch with steady states labeled
        (stable = black, unstable = red) """
        xs = np.linspace(0, 1, 1001)
        x_nullcline = np.array([-(s.mu[0] + x*s.M[0][0])/s.M[0][1] for x in xs])
        y_nullcline = np.array([-(s.mu[1] + x*s.M[1][0])/s.M[1][1] for x in xs])

        fps = [s.ssa, s.get_mixed_fp(), s.ssb]
        is_stables = s.get_fp_stability(fps)

        for fp,is_stable in zip(fps, is_stables):
            print(fp)
            color = 'k' if is_stable else 'r'
            ax.plot(fp[0], fp[1], c=color, marker='.', ms=25, zorder=4)

        ax.plot(xs, x_nullcline, c='grey', label='$\dot{x} = 0$')
        ax.plot(xs, y_nullcline, c='grey', label='$\dot{y} = 0$')
        ax.legend(loc='upper right'); ax.set_xlabel('$u$'); ax.set_ylabel('$v$')
        ax.set_title('$\\alpha =${:.3}, $\\beta = ${:.3}, and $\gamma = ${:.3} (gLV basis)'
                     .format(s.alpha, s.beta, s.gamma), fontsize=18)
        #ax.axis([0, max(y_nullcline[:,0]), 0, max(x_nullcline[:,1])])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        filename = 'figs/sp_nullclines_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))

    def plot_example_traj(s, ax, savefig=True):
        """ Plot an identical sample trajectory as the one in the
        originalParams class"""
        #op_ic = np.array([1.8, 3])
        op_ic = np.array([.1, .3])
        qp = quasipolynomialParams()
        gp = gLVParams(qp)
        qp_ic = qp.convert_op_to_qp([op_ic])[0]
        gp_ic = gp.convert_qp_to_gp([qp_ic])[0]
        sp_ic = s.convert_gp_to_sp([gp_ic])[0]
        print('SSR IC is {}'.format(sp_ic))

        t = np.linspace(0, 5, 501)
        z = integrate.odeint(s.get_integrand, sp_ic, t)
        if False:
            print('gp trajectory:')
            print(z[:, :2])
        #x = np.array([(zz[3]*zz[4]*zz[5]/(zz[2]**2))**(1/3) for zz in z])
        #y = np.array([(zz[2]*zz[3]*zz[5]/(zz[4]**2))**(1/3) for zz in z])
        ax.plot(z[:,0], z[:,1], '-g', zorder=2, label='SSR form')
        ax.legend()
        print(z[-1])
        #print()
        #print(s.get_jacobian(z[-1]))
        #print(np.linalg.eig(s.get_jacobian(z[-1])))
        #ax.axis([0, 1, 0, 1])

        filename = 'figs/sp_example_traj_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))

# from Ian Hincks, https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])

def compare_trajectories(op, sp):
    """ Plot a sample trajectory in both originalParams and in ssrParams"""
    fig, ax = plt.subplots(figsize=(6,6))
    op.plot_nullclines(ax, savefig=False)

    for ic in [[.5, .5], [1.4, 3], [1, 4], [4, 1], [2, 3], [.35, 1.2]]:
        op_ic = np.array(ic)
        sp_ic = sp.convert_op_to_sp([op_ic])[0]
        #print('originalParams IC is {}'.format(op_ic))
        #print('ssrParams IC is {}'.format(sp_ic))

        t = np.linspace(0, 100, 501)
        orig_z = integrate.odeint(op.get_integrand, op_ic, t)
        ssr_z = integrate.odeint(sp.get_integrand, sp_ic, t)
        orig_z_ssr_basis = sp.convert_op_to_sp(orig_z)
        ssr_z_orig_basis = op.convert_sp_to_op(ssr_z)

        #print(orig_z)
        #print(ssr_z_orig_basis)
        #print(ssr_z)
        #print(orig_z_ssr_basis)

        ax.plot(op_ic[0], op_ic[1], 'k.', ms=17)
        ax1, = ax.plot(orig_z[:,0], orig_z[:,1], color='green', ls='--', zorder=2, label='original form')
        ax2, = ax.plot(ssr_z_orig_basis[:,0], ssr_z_orig_basis[:,1],
                       color='purple', ls='-.', zorder=2, label='SSR form')
        ax.legend([ax1, ax2], ['original form', 'SSR form'])
        ax.axis([0, None, 0, None])

    filename = 'figs/orig_basis_traj_{}.pdf'.format(op.ext)
    plt.savefig(filename, bbox_inches='tight')
    print('... saved to {}'.format(filename))

    fig, ax = plt.subplots(figsize=(6,6))
    sp.plot_nullclines(ax, savefig=False)

    for ic in [[.5, .5], [1.4, 3], [1, 4], [4, 1], [2, 3], [.35, 1.2]]:
        op_ic = np.array(ic)
        sp_ic = sp.convert_op_to_sp([op_ic])[0]
        #print('originalParams IC is {}'.format(op_ic))
        #print('ssrParams IC is {}'.format(sp_ic))

        t = np.linspace(0, 100, 501)
        orig_z = integrate.odeint(op.get_integrand, op_ic, t)
        ssr_z = integrate.odeint(sp.get_integrand, sp_ic, t)
        orig_z_ssr_basis = sp.convert_op_to_sp(orig_z)
        ssr_z_orig_basis = op.convert_sp_to_op(ssr_z)

        #print(orig_z)
        #print(ssr_z_orig_basis)
        #print(ssr_z)
        #print(orig_z_ssr_basis)

        ax.plot(sp_ic[0], sp_ic[1], 'k.', ms=17)
        ax1, = ax.plot(ssr_z[:,0], ssr_z[:,1], color='green', ls='--', zorder=2, label='original form')
        ax2, = ax.plot(orig_z_ssr_basis[:,0], orig_z_ssr_basis[:,1],
                       color='purple', ls='-.', zorder=2, label='SSR form')
        ax.legend([ax1, ax2], ['original form', 'SSR form'])
        ax.axis([0, None, 0, None])

    filename = 'figs/ssr_basis_traj_{}.pdf'.format(op.ext)
    plt.savefig(filename, bbox_inches='tight')
    print('... saved to {}'.format(filename))

def compare_steady_states(op, sp, savefig=True):
    phase_orig = {}
    phase_ssr = {}
    max_x = max(op.ssa[0], op.ssb[0])
    max_y = max(op.ssa[1], op.ssb[1])
    num_points = 51 # 11, 31, 51 saved
    xs = np.linspace(0, max_x*1.1, num_points)
    ys = np.linspace(0, max_y*1.1, num_points)

    filename = 'bistable_switch_phases_{}'.format(num_points)
    read_data = True
    if not read_data:
        for x in xs:
            print(x)
            for y in ys:
                op_ic = np.array([x, y])
                sp_ic = sp.convert_op_to_sp([op_ic])[0]

                t = np.linspace(0, 200, 501)
                orig_z = integrate.odeint(op.get_integrand, op_ic, t)
                ssr_z = integrate.odeint(sp.get_integrand, sp_ic, t)
                ssr_z_orig_basis = op.convert_sp_to_op(ssr_z)
                orig_z_ssr_basis = sp.convert_op_to_sp(orig_z)

                eps = .001
                for z, form in zip([orig_z, ssr_z], [op, sp]):
                    if np.linalg.norm(z[-1] - form.ssa) < eps: color = 'purple'
                    elif np.linalg.norm(z[-1] - form.ssb) < eps: color = 'g'
                    else: print('{}, {}: neither SS'.format(x, y)); color = 'orange'

                    if form == op:
                        phase_orig[(x, y)] = (orig_z[-1], color)
                    if form == sp:
                        phase_ssr[tuple(sp_ic)] = (ssr_z[-1], color)
        with open('data/{}'.format(filename), 'wb') as f:
            pickle.dump((phase_orig, phase_ssr), f)
            print('... SAVED data to {}'.format(filename))
    else:
        with open('data/{}'.format(filename), 'rb') as f:
            phase_orig, phase_ssr = pickle.load(f)
            print('... LOADED data from {}'.format(filename))

    fig, ax = plt.subplots(figsize=(6,6))
    op.plot_nullclines(ax, savefig=False)

    for phase, name in zip([phase_orig, phase_ssr], ['orig', 'ssr']):
        greens = []
        purples = []
        for key in phase:
            if name is 'orig':
                if phase[key][1] == 'g':
                    greens.append(list(key))
                elif phase[key][1] == 'purple':
                    purples.append(list(key))
                else:
                    print('{} didn\'t go to either steady state'.format(key))
            elif name is 'ssr':
                if phase[key][1] == 'g':
                    greens.append(op.convert_sp_to_op([list(key)])[0])
                elif phase[key][1] == 'purple':
                    purples.append(op.convert_sp_to_op([list(key)])[0])
                else:
                    print('{} didn\'t go to either steady state'.format(key))

        greens = np.array(greens); purples = np.array(purples)
        if name is 'orig':
            markertype = 's'
            green = lighten_color('green', .3)
            purple = lighten_color('purple', .3)
            zorder = 0
            markersize = 5.3
            alpha = 1
        elif name is 'ssr':
            markertype = 'D'
            green = lighten_color('green', 1)
            purple = lighten_color('purple', 1)
            zorder = 1
            markersize = 3
            alpha = 1

        ax.plot(greens[:,0], greens[:,1], markertype, color=green,
                zorder=zorder, markersize=markersize, alpha=alpha)
        ax.plot(purples[:,0], purples[:,1], markertype, color=purple, zorder=zorder,
                markersize=markersize, alpha=alpha)

        ## make custom legend handler (from Joel @
        ## https://stackoverflow.com/questions/31478077/how-to-make-two-markers-share-the-same-label-in-the-legend-using-matplotlib )
        class AnyObject(object):
            pass

        class data_handler(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                box_greens = ['green', lighten_color('green', .3)]
                box_purples = ['purple', lighten_color('purple', .3)]
                scale = fontsize / 13
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                interval = width/2
                patch_sq = mpl.patches.Rectangle([x0, y0 + height/2 * (1 - scale) ], height * scale,
                               height * scale, facecolor=box_greens[1],
                               transform=handlebox.get_transform())
                patch_circ = mpl.patches.Rectangle([x0 + interval, y0
                    + height/2*(1-scale)], height * scale, height*scale,
                    facecolor=box_purples[1],
                    transform=handlebox.get_transform())

                handlebox.add_artist(patch_sq)
                handlebox.add_artist(patch_circ)
                return True

        class AnyObject2(object):
            pass

        class data_handler2(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                box_greens = ['green', lighten_color('green', .3)]
                box_purples = ['purple', lighten_color('purple', .3)]
                scale = fontsize / 13
                x0, y0 = handlebox.xdescent + 5, handlebox.ydescent + 5
                width, height = handlebox.width, handlebox.height
                interval = width/2
                patch_sq = mpl.patches.RegularPolygon([x0, y0 + height/2 * (1 -
                    scale)], numVertices=4, radius=5, orientation=0,
                    facecolor=box_greens[0],
                    transform=handlebox.get_transform())
                patch_circ = mpl.patches.RegularPolygon([x0 + interval, y0 +
                    height/2*(1-scale)], numVertices=4, radius=5,
                    orientation=0, facecolor=box_purples[0],
                    transform=handlebox.get_transform())
                ## https://stackoverflow.com/questions/4285103/matplotlib-rotating-a-patch
                #t2 = mpl.transforms.Affine2D().rotate_deg(-45) + handlebox.get_transform()
                #patch_sq.set_transform(t2)


                handlebox.add_artist(patch_sq)
                handlebox.add_artist(patch_circ)
                return True

    leg = ax.legend([AnyObject(), AnyObject2(), mpl.lines.Line2D([0], [0],
                     color='dimgrey')],
                    ['true basin of attraction', 'SSR basin of attraction',
                     'true nullclines'],
                    handler_map={AnyObject: data_handler(), AnyObject2:
                                 data_handler2()},
                    fontsize=14, loc='upper right', handlelength=2,
                    framealpha=1)

    ax.set_xlim(0, max_x*1.1)
    ax.set_ylim(0, max_y*1.1)

    if savefig:
        plt.savefig('figs/{}_{}.pdf'.format(filename, op.ext), bbox_inches='tight')
        print('... saved to {}_{}.pdf'.format(filename, op.ext))


def verify_all_fps_are_fps():
    """ Test that fixed points of the originalParams translate to also being
    fixed points of the quasipolynomialParams and gLVParams """
    print('TESTING fixed points are preserved in all algebraic forms')
    op = originalParams()
    sp = ssrParams(op)
    fps = op.get_stable_fps()
    sp_fps = sp.convert_op_to_sp(fps)
    verbose = True
    for op_fp,sp_fp in zip(fps, sp_fps):
        if verbose:
            print('    originalParams f.p. {} integrand: {}'.format(op_fp,op.get_integrand(op_fp, 0)))
            print('    ssrParams f.p. {} integrand: {}'.format(sp_fp,sp.get_integrand(sp_fp, 0)))

def verify_all_fps_are_stable():
    """ Test that the fixed points are stable in each system """
    print('TESTING fixed points are stable in all forms')
    op = originalParams()
    sp = ssrParams(op)

    for p,p_name in zip([op, sp], ['originalParams', 'ssrParams']):
        print('    {} fps are {} and {} with eigenvalues:'.format(p_name, p.fps[0], p.fps[1]))
        for fp in p.fps:
            print('      {}'.format(np.linalg.eig(p.get_jacobian(fp))[0]))

def verify_all_trajs_are_same():
    """ Plot the same trajectory in the three different forms (op, qp, gp) """
    print('TESTING all algebraic forms lead to same trajectory')
    op = originalParams()
    sp = ssrParams(op)
    fig, ax = plt.subplots(figsize=(6,6))
    op.plot_nullclines(ax, savefig=False)
    sp.plot_example_traj(ax, savefig=True)


def sanity_checks():
    """ Run sanity check tests that ensure that algebraic manipulations don't
    change trajectories/fixed points/stability """
    verify_all_fps_are_fps()
    print('----')
    verify_all_fps_are_stable()
    #print('----')
    #verify_all_trajs_are_same()

if __name__ == '__main__':
    #sanity_checks()

    op = originalParams()
    sp = ssrParams(op)
    compare_steady_states(op, sp)

    #print('----')

    #print('originalParams steady states: {}, {}'.format(op.ssa, op.ssb))
    #print('ssrParams steady states: {}, {}'.format(sp.ssa, sp.ssb))
    #print('ssrParams mu and M:')
    #print(sp.mu)
    #print(sp.M)
    #fig, ax = plt.subplots(figsize=(6,6))
    #x_null, y_null = op.get_nullclines()
    #true_fps = op.get_intersections(x_null, y_null)
    #print(true_fps)
