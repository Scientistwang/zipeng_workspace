#!/usr/bin/env python3
#
# Reduction of a gene regulatory network (the bistable switch) into a
# steady-state reduced form, where the two stable states of the gene circuit
# correspond to the steady states (1, 0) and (0, 1) in the reduced
# 2-dimensional generalized Lotka-Volterra system.


import numpy as np
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt

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
        return np.array([s.alpha/(1 + Z[1]**2) - Z[0],
                         s.beta/(1 + Z[0]**2) - s.gamma*Z[1]])

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

        for fp,is_stable in zip(fps, is_stables):
            color = 'k' if is_stable else 'r'
            ax.plot(fp[0], fp[1], c=color, marker='.', ms=25, zorder=4)

        ax.plot(x_nullcline[:,0], x_nullcline[:,1], label='$\dot{x} = 0$')
        ax.plot(y_nullcline[:,0], y_nullcline[:,1], label='$\dot{y} = 0$')
        ax.legend(); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('Nullclines with $\\alpha =${:.3}, $\\beta = ${:.3}, and $\gamma = ${:.3}'
                     .format(s.alpha, s.beta, s.gamma), fontsize=18)
        ax.axis([0, max(y_nullcline[:,0]), 0, max(x_nullcline[:,1])])
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 1, 2, 3, 4])

        filename = 'figs/op_nullclines_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))

    def plot_example_traj(s, ax, savefig=True):
        ic = np.array([1.8, 3])
        t = np.linspace(0, 500, 501)
        y = integrate.odeint(s.get_integrand, ic, t)
        if False:
            print('orig params trajectory:')
            print(y)
        ax.plot(y[:,0], y[:,1], '--k', zorder=3)

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
        stab_fps = np.array(stab_fps)

        return stab_fps

class quasipolynomialParams:
    """ This class contains all methods relevant for the quasipolynomial
    differential equation form of the bistable switch, which consists of 8
    coupled ODEs, each of which may be up to fourth order. """
    def __init__(s, op=originalParams()):
        s.alpha, s.beta, s.gamma = op.alpha, op.beta, op.gamma
        s.ext = 'a{:.3}_b{:.3}_g{:.3}'.format(s.alpha, s.beta, s.gamma)

        s.ssa = s.convert_op_to_qp([op.ssa])[0]
        s.ssb = s.convert_op_to_qp([op.ssb])[0]
        s.fps = np.array([s.ssa, s.ssb])
        #print(s.ssa[4], s.ssa[5], s.ssa[6], s.ssa[7])
        #print(s.ssa[4]*s.ssa[6])
        #print(s.ssa[5]*s.ssa[7])

    def get_integrand(s, Y, t):
        """ Synthetic gene circuit for a bistable switch in quasipolynomial
        form. To "simplify" this equation back to the original 2-dimensional
        ODE form, set a=1+z^2, b=1+y^2, c=a^-1, d=b^-1, e=y^-1, and f=z^-1.
        This integrand returns the derivative of the vector 
        [y, z, a, b, c, d, e, f]. """
        return np.array([Y[0]*(s.alpha*Y[4]*Y[6] - 1),
                         Y[1]*(s.beta*Y[5]*Y[7] - s.gamma),
                         Y[2]*(2*s.beta*Y[1]*Y[4]*Y[5] - 2*s.gamma*Y[1]*Y[1]*Y[4]),
                         Y[3]*(2*s.alpha*Y[0]*Y[4]*Y[5] - 2*Y[0]*Y[0]*Y[5]),
                         Y[4]*(-2*s.beta*Y[1]*Y[4]*Y[5] + 2*s.gamma*Y[1]*Y[1]*Y[4]),
                         Y[5]*(-2*s.alpha*Y[0]*Y[4]*Y[5] + 2*Y[0]*Y[0]*Y[5]),
                         Y[6]*(-s.alpha*Y[4]*Y[6] + 1),
                         Y[7]*(-s.beta*Y[5]*Y[7] + s.gamma)])

    def get_jacobian(s, fp):
        """ Return the jacobian of the bistable switch in quasipolynomial
        coordinates evaluated at the fixed point fp. Variables are [y, z, a, b,
        c, d, e, f]. """
        Y = fp
        jac = np.array(
              [[s.alpha*Y[4]*Y[6] - 1, 0, 0, 0, s.alpha*Y[0]*Y[6], 0,
                   s.alpha*Y[0]*Y[4], 0],
               [0, s.beta*Y[5]*Y[7] - s.gamma, 0, 0, 0, s.beta*Y[1]*Y[7],
                   0, s.beta*Y[1]*Y[5]],
               [0, 2*s.beta*Y[2]*Y[4]*Y[5] - 4*s.gamma*Y[1]*Y[2]*Y[4],
                   2*s.beta*Y[1]*Y[4]*Y[5] - 2*s.gamma*Y[1]*Y[1]*Y[4], 0,
                   2*s.beta*Y[1]*Y[2]*Y[5] - 2*s.gamma*Y[1]*Y[1]*Y[2],
                   2*s.beta*Y[1]*Y[2]*Y[4], 0, 0],
               [2*s.alpha*Y[3]*Y[4]*Y[5] - 4*Y[0]*Y[3]*Y[5], 0, 0,
                   2*s.alpha*Y[0]*Y[4]*Y[5] - 2*Y[0]*Y[0]*Y[5],
                   2*s.alpha*Y[0]*Y[3]*Y[5],
                   2*s.alpha*Y[0]*Y[3]*Y[4] - 2*Y[0]*Y[0]*Y[3], 0, 0],
               [0, -2*s.beta*Y[4]*Y[4]*Y[5] + 4*s.gamma*Y[1]*Y[4]*Y[4], 0, 0,
                   -4*s.beta*Y[1]*Y[4]*Y[5] + 4*s.gamma*Y[1]*Y[1]*Y[4],
                   -2*s.beta*Y[1]*Y[4]*Y[4], 0, 0],
               [-2*s.alpha*Y[4]*Y[5]*Y[5] + 4*Y[0]*Y[5]*Y[5], 0, 0, 0,
                   -2*s.alpha*Y[0]*Y[5]*Y[5],
                   -4*s.alpha*Y[0]*Y[4]*Y[5] + 4*Y[0]*Y[0]*Y[5], 0, 0],
               [0, 0, 0, 0, -s.alpha*Y[6]*Y[6], 0,
                   -2*s.alpha*Y[4]*Y[6] + 1, 0],
               [0, 0, 0, 0, 0, -s.beta*Y[7]*Y[7], 0,
                   -2*s.beta*Y[5]*Y[7] + s.gamma]])
        return jac

    def convert_op_to_qp(s, points):
        """ Take a list of 2D points (corresponding to originalParams) and
        output a list of the corresponding 8D points in the quasipolynomial
        basis ([y, z, a, b, c, d, e, f]). Recall that a=1+z^2, b=1+y^2, c=a^-1,
        d=b^-1, e=y^-1, and f=z^-1. """
        new_points = []
        for point in points:
            y = point[0]
            z = point[1]
            new = [y, z, 1+z**2, 1+y**2, 1/(1+z**2), 1/(1+y**2), 1/y, 1/z]
            new_points.append(new)
        new_points = np.array(new_points)
        return new_points

    def plot_example_traj(s, ax, savefig=True):
        """ Plot an identical sample trajectory as the one in the
        originalParams class"""
        op_ic = np.array([1.8, 3])
        qp_ic = s.convert_op_to_qp([op_ic])[0]

        t = np.linspace(0, 500, 501)
        y = integrate.odeint(s.get_integrand, qp_ic, t)
        if False:
            print('qp trajectory:')
            print(y[:, :2])
        ax.plot(y[:,0], y[:,1], ':b', zorder=3)

        filename = 'figs/qp_example_traj_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))

class gLVParams:
    """ This class contains all methods relevant for the gLV 
    differential equation form of the bistable switch, which consists of 6
    coupled ODEs, each of which may be up to second order. """
    def __init__(s, qp=quasipolynomialParams()):
        s.alpha, s.beta, s.gamma = qp.alpha, qp.beta, qp.gamma

        s.mu = np.array([1, s.gamma, -s.gamma, -2*s.gamma, -1, -2])
        s.M = np.array([[-s.alpha, 0, -2*s.beta, 2*s.gamma, 0, 0],
                        [0, -s.beta, 0, 0, -2*s.alpha, 2],
                        [0, s.beta, -2*s.beta, 2*s.gamma, -2*s.alpha, 2],
                        [0, 2*s.beta, -2*s.beta, 2*s.gamma, 0, 0],
                        [s.alpha, 0, -2*s.beta, 2*s.gamma, -2*s.alpha, 2],
                        [2*s.alpha, 0, 0, 0, -2*s.alpha, 2]])

        s.ssa = s.convert_qp_to_gp([qp.ssa])[0]
        s.ssb = s.convert_qp_to_gp([qp.ssb])[0]
        s.fps = np.array([s.ssa, s.ssb])
        #print(s.ssa)
        #print(s.ssb)

        s.ext = 'a{:.3}_b{:.3}_g{:.3}'.format(s.alpha, s.beta, s.gamma)

    def get_integrand(s, Y, t):
        """ Synthetic gene circuit for a bistable switch in gLV
        form. To "simplify" this equation back to the quasipolynomial
        form, set g=ce, h=df, l=zdc, m=z^2 c, n=ycd, p=y^2 d.
        This integrand returns the derivative of the vector 
        [g, h, l, m, n, p]. """
        return np.dot(np.diag(s.mu), Y) + np.dot( np.diag(np.dot(s.M, Y)), Y)

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

    def convert_qp_to_gp(s, points):
        """ Take a list of 8D points (corresponding to quasipolynomialParams) and
        output a list of the corresponding 6D points in the gLV
        basis ([g, h, l, m, n, p]). Recall that g=ce, h=df, l=zdc, m=z^2 c,
        n=ycd, p=y^2 d. """
        new_points = []
        for point in points:
            y, z, a, b, c, d, e, f = point
            new = [c*e, d*f, z*d*c, z*z*c, y*c*d, y*y*d]
            new_points.append(new)
        new_points = np.array(new_points)
        return new_points

    def plot_example_traj(s, ax, savefig=True):
        """ Plot an identical sample trajectory as the one in the
        originalParams class"""
        op_ic = np.array([1.8, 3])
        qp = quasipolynomialParams()
        qp_ic = qp.convert_op_to_qp([op_ic])[0]
        gp_ic = s.convert_qp_to_gp([qp_ic])[0]

        t = np.linspace(0, 500, 501)
        z = integrate.odeint(s.get_integrand, gp_ic, t)
        if False:
            print('gp trajectory:')
            print(z[:, :2])
        x = np.array([(zz[3]*zz[4]*zz[5]/(zz[2]**2))**(1/3) for zz in z])
        y = np.array([(zz[2]*zz[3]*zz[5]/(zz[4]**2))**(1/3) for zz in z])
        ax.plot(x, y, '-.r', zorder=2)

        filename = 'figs/gp_example_traj_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))


class ssrParams:
    """ This class contains all methods relevant for the steady state reduction
    of the gLV differential equation form of the bistable switch. """
    def __init__(s, gp=gLVParams()):

        s.gp_ssa = gp.ssa
        s.gp_ssb = gp.ssb

        #print(gp.ssa)
        #print(gp.ssb)

        s.ssa = s.convert_gp_to_sp([s.gp_ssa])[0]
        s.ssb = s.convert_gp_to_sp([s.gp_ssb])[0]
        s.fps = np.array([s.ssa, s.ssb])

        s.mu = np.array([np.dot(gp.mu, gp.ssa), np.dot(gp.mu, gp.ssb)])
        s.M = np.array([[np.dot(gp.ssa.T, np.dot(gp.M, gp.ssa)), np.dot(gp.ssa.T, np.dot(gp.M, gp.ssb))],
                        [np.dot(gp.ssb.T, np.dot(gp.M, gp.ssa)), np.dot(gp.ssb.T, np.dot(gp.M, gp.ssb))]])
        #print(gp.mu)
        #print(gp.ssa)
        #print(s.mu)
        #print(s.M)
        #else:
        #    s.alpha = 4.09
        #    s.beta = .73
        #    s.gamma = .2
        #s.ext = 'a{:.3}_b{:.3}_g{:.3}'.format(s.alpha, s.beta, s.gamma)

    def get_integrand(s, Y, t):
        """ Synthetic gene circuit for a bistable switch in gLV
        form. To "simplify" this equation back to the quasipolynomial
        form, set g=ce, h=df, l=zdc, m=z^2 c, n=ycd, p=y^2 d.
        This integrand returns the derivative of the vector 
        [g, h, l, m, n, p]. """
        return np.array(
                [Y[0]*(1 - s.alpha*Y[0] - 2*s.beta*Y[2] + 2*s.gamma*Y[3]),
                 Y[1]*(s.gamma - s.beta*Y[1] - 2*s.alpha*Y[4] + 2*Y[5]),
                 Y[2]*(-s.gamma + s.beta*Y[1] - 2*s.beta*Y[2] + 2*s.gamma*Y[3]
                       - 2*s.alpha*Y[4] + 2*Y[5]),
                 Y[3]*(-2*s.gamma + 2*s.beta*Y[1] - 2*s.beta*Y[2]
                       + 2*s.gamma*Y[3]),
                 Y[4]*(-1 + s.alpha*Y[0] - 2*s.beta*Y[2] + 2*s.gamma*Y[3]
                       - 2*s.alpha*Y[4] + 2*Y[5]),
                 Y[5]*(-2 + 2*s.alpha*Y[0] - 2*s.alpha*Y[4] + 2*Y[5])]) 

    def convert_gp_to_sp(s, points):
        """ Take a list of 6D points (corresponding to gLVParams) and
        output a list of the corresponding 2D points in the SSR
        basis ([x, y]). """
        new_points = []
        for point in points:
            uu = np.dot(s.gp_ssa, s.gp_ssa); vv = np.dot(s.gp_ssb, s.gp_ssb)
            xu = np.dot(point, s.gp_ssa); xv = np.dot(point, s.gp_ssb)
            uv = np.dot(s.gp_ssa, s.gp_ssb)
            new_points.append([(xu*vv - xv*uv)/(uu*vv - uv**2),
                               (uu*xv - xu*uv)/(uu*vv - uv**2)])
        new_points = np.array(new_points)
        return new_points

    def plot_example_traj(s, ax, savefig=True):
        """ Plot an identical sample trajectory as the one in the
        originalParams class"""
        op_ic = np.array([1.8, 3])
        qp = quasipolynomialParams()
        qp_ic = qp.convert_op_to_qp([op_ic])[0]
        gp_ic = s.convert_qp_to_gp([qp_ic])[0]

        t = np.linspace(0, 500, 501)
        z = integrate.odeint(s.get_integrand, gp_ic, t)
        if False:
            print('gp trajectory:')
            print(z[:, :2])
        x = np.array([(zz[3]*zz[4]*zz[5]/(zz[2]**2))**(1/3) for zz in z])
        y = np.array([(zz[2]*zz[3]*zz[5]/(zz[4]**2))**(1/3) for zz in z])
        ax.plot(x, y, '-.r', zorder=2)

        filename = 'figs/gp_example_traj_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))


def verify_all_fps_are_fps():
    """ Test that fixed points of the originalParams translate to also being
    fixed points of the quasipolynomialParams and gLVParams """
    op = originalParams()
    qp = quasipolynomialParams(op)
    gp = gLVParams(qp)
    fps = op.get_stable_fps()
    qp_fps = qp.convert_op_to_qp(fps)
    gp_fps = gp.convert_qp_to_gp(qp_fps)
    for op_fp,qp_fp,gp_fp in zip(fps, qp_fps, gp_fps):
        print(op.get_integrand(op_fp, 0))
        print(qp.get_integrand(qp_fp, 0))
        print(gp.get_integrand(gp_fp, 0))

def verify_all_fps_are_stable():
    """ Test that the fixed points are stable in each system """
    op = originalParams()
    qp = quasipolynomialParams(op)
    gp = gLVParams(qp)

    for p in [op, qp, gp]:
        print(p.fps)
        for fp in p.fps:
            print(np.linalg.eig(p.get_jacobian(fp))[0])
        print()

def verify_all_trajs_are_same():
    """ Plot the same trajectory in the three different forms (op, qp, gp) """
    op = originalParams()
    qp = quasipolynomialParams(op)
    gp = gLVParams(qp)
    fig, ax = plt.subplots(figsize=(6,6))
    op.plot_nullclines(ax, savefig=False)
    op.plot_example_traj(ax, savefig=False)
    qp.plot_example_traj(ax, savefig=False)
    gp.plot_example_traj(ax)

if __name__ == '__main__':
    #verify_all_fps_are_fps()
    #verify_all_trajs_are_same()
    verify_all_fps_are_stable()

    #op = originalParams()
    #qp = quasipolynomialParams(op)
    #gp = gLVParams(qp)
    #sp = ssrParams(gp)

    #fig, ax = plt.subplots(figsize=(6,6))
    #x_null, y_null = op.get_nullclines()
    #true_fps = op.get_intersections(x_null, y_null)
    #print(true_fps)
