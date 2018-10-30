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
np.set_printoptions(suppress=True, precision=4)

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
        ax.legend(); ax.set_xlabel('gene $x$ concentration'); ax.set_ylabel('gene $y$ concentration')
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
        ODE form, set u=1/y, v=1/z, a=1/(1+v^2), and b=1/(1+u^2).  This
        integrand returns the derivative of the vector [u, v, a, b] """
        return np.array([Y[0]*(1 - s.alpha*Y[0] + s.alpha*Y[0]*Y[2]),
                         Y[1]*(s.gamma - s.beta*Y[1] + s.beta*Y[1]*Y[3]),
                         Y[2]*(-2*s.gamma*Y[1]**2*Y[2] + 2*s.beta*Y[1]**3*Y[2]
                               - 2*s.beta*Y[1]**3*Y[2]*Y[3]),
                         Y[3]*(-2*Y[0]**2*Y[3] + 2*s.alpha*Y[0]**3*Y[3]
                               - 2*s.alpha*Y[0]**3*Y[2]*Y[3])])

    def get_jacobian(s, fp):
        """ Return the jacobian of the bistable switch in quasipolynomial
        coordinates evaluated at the fixed point fp. Variables are [y, z, a, b,
        c, d, e, f]. """
        u,v,a,b = fp
        jac = np.array(
            [[1 - 2*s.alpha*u + 2*s.alpha*u*a, 0, s.alpha*u**2, 0],
             [0, s.gamma - 2*s.beta*v + 2*s.beta*v*b, 0, s.beta*v**2],
             [0, -4*s.gamma*a**2*v + 6*s.beta*a**2*v**2 - 6*s.beta*a**2*v**2*b,
              -4*s.gamma*a*v**2 + 4*s.beta*a*v**3 - 4*s.beta*a*b*v**3,
              -2*s.beta*a**2*v**3],
             [-4*b**2*u + 6*s.alpha*b**2*u**2 - 6*s.alpha*a*b**2*u**2, 0,
              -2*s.alpha*b**2*u**3, -4*b*u**2 + 4*s.alpha*b*u**3 -
              4*s.alpha*a*b*u**3]])
        return jac

    def convert_op_to_qp(s, points):
        """ Take a list of 2D points (corresponding to originalParams) and
        output a list of the corresponding 8D points in the quasipolynomial
        basis [u, v, a, b]. Recall that u=1/y, v=1/z, a=1/(1+v^2), and
        b=1/(1+u^2) """
        new_points = []
        for point in points:
            y = point[0]
            z = point[1]
            u = 1/y
            v = 1/z
            new = [u, v, 1/(1+v**2), 1/(1+u**2)]
            new_points.append(new)
        new_points = np.array(new_points)
        return new_points

    def plot_example_traj(s, ax, savefig=True):
        """ Plot an identical sample trajectory as the one in the
        originalParams class"""
        op_ic = np.array([1.8, 3])
        qp_ic = s.convert_op_to_qp([op_ic])[0]

        t = np.linspace(0, 500, 501)
        z = integrate.odeint(s.get_integrand, qp_ic, t)
        if False:
            print('qp trajectory:')
            print(z[:, :2])
        x = np.array([1/zz[0] for zz in z])
        y = np.array([1/zz[1] for zz in z])
        ax.plot(x, y, ':b', zorder=3, label='QP form')
        ax.legend(fontsize=12)

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

        s.mu = np.array([1, s.gamma, 1, s.gamma, 2*s.gamma, 3*s.gamma,
                         3*s.gamma, 2, 3, 3])
        s.M = np.array(
               [[-s.alpha, 0, s.alpha, 0, 0, 0, 0, 0, 0, 0],
                [0, -s.beta, 0, s.beta, 0, 0, 0, 0, 0, 0],
                [-s.alpha, 0, s.alpha, 0, -2*s.gamma, 2*s.beta, -2*s.beta, 0,
                 0, 0],
                [0, -s.beta, 0, s.beta, 0, 0, 0, -2, 2*s.alpha, -2*s.alpha],
                [0, -2*s.beta, 0, 2*s.beta, -2*s.gamma, 2*s.beta, -2*s.beta, 0,
                 0, 0],
                [0, -3*s.beta, 0, 3*s.beta, -2*s.gamma, 2*s.beta, -2*s.beta, 0,
                 0, 0],
                [0, -3*s.beta, 0, 3*s.beta, -2*s.gamma, 2*s.beta, -2*s.beta, -2,
                 2*s.alpha, -2*s.alpha],
                [-2*s.alpha, 0, 2*s.alpha, 0, 0, 0, 0, -2, 2*s.alpha,
                 -2*s.alpha],
                [-3*s.alpha, 0, 3*s.alpha, 0, 0, 0, 0, -2, 2*s.alpha,
                 -2*s.alpha],
                [-3*s.alpha, 0, 3*s.alpha, 0, -2*s.gamma, 2*s.beta, -2*s.beta,
                 -2, 2*s.alpha, -2*s.alpha]])

        s.ssa = s.convert_qp_to_gp([qp.ssa])[0]
        s.ssb = s.convert_qp_to_gp([qp.ssb])[0]
        s.fps = np.array([s.ssa, s.ssb])
        #print(s.ssa)
        #print(s.ssb)

        s.ext = 'a{:.3}_b{:.3}_g{:.3}'.format(s.alpha, s.beta, s.gamma)

    def get_integrand(s, Y, t):
        """ Synthetic gene circuit for a bistable switch in gLV
        form. To "simplify" this equation back to the quasipolynomial
        form, set c=u*a, d=v*b, e=v^2*a, f=v^3*a, g=v^3*a*b, h=u^2*b, k=u^3*b,
        l=u^3*a*b.  This integrand returns the derivative of the vector
        [u,v,c,d,e,f,g,h,k,l]. """
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
        """ Take a list of 8D points (corresponding to quasipolynomialParams)
        and output a list of the corresponding 6D points in the gLV basis
        ([u,v,c,d,e,f,g,h,k,l]). Recall that c=u*a, d=v*b, e=v^2*a, f=v^3*a,
        g=v^3*a*b, h=u^2*b, k=u^3*b, l=u^3*a*b"""
        new_points = []
        for point in points:
            u,v,a,b = point
            new = [u, v, u*a, v*b, v**2*a, v**3*a, v**3*a*b, u**2*b, u**3*b,
                   u**3*a*b]
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
        x = np.array([1/zz[0] for zz in z])
        y = np.array([1/zz[1] for zz in z])
        ax.plot(x, y, '-.r', zorder=2, label='gLV form')
        ax.legend()

        filename = 'figs/gp_example_traj_{}.pdf'.format(s.ext)
        if savefig:
            plt.savefig(filename, bbox_inches='tight')
            print('... saved to {}'.format(filename))


class ssrParams:
    """ This class contains all methods relevant for the steady state reduction
    of the gLV differential equation form of the bistable switch. """
    def __init__(s, gp=gLVParams()):

        s.alpha, s.beta, s.gamma = gp.alpha, gp.beta, gp.gamma
        s.gp_ssa = gp.ssa
        s.gp_ssb = gp.ssb

        print('gLV steady states:')
        print(gp.ssa)
        print(gp.ssb)
        print()

        s.ssa = s.convert_gp_to_sp([s.gp_ssa])[0]
        s.ssb = s.convert_gp_to_sp([s.gp_ssb])[0]
        s.fps = np.array([s.ssa, s.ssb])

        s.mu = np.array([np.dot(gp.mu, gp.ssa), np.dot(gp.mu, gp.ssb)])
        s.M = np.array([[np.dot(gp.ssa, np.dot(gp.M, gp.ssa)), np.dot(gp.ssa, np.dot(gp.M, gp.ssb))],
                        [np.dot(gp.ssb, np.dot(gp.M, gp.ssa)), np.dot(gp.ssb, np.dot(gp.M, gp.ssb))]])
        print('extra:')
        print(gp.ssa)
        print(gp.ssb)
        print(np.dot(gp.M, gp.ssa))
        print(np.dot(gp.M, gp.ssb))
        print()
        print('gLV mu and M:')
        print(gp.mu)
        a = gp.M.copy()
        a[[1, 2], :] = a[[2, 1], :]
        a[:, [1, 2]] = a[:, [2, 1]]
        print(gp.M)
        print()
        print('SSR mu and M:')
        print(s.mu)
        print(s.M)
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


def verify_all_fps_are_fps():
    """ Test that fixed points of the originalParams translate to also being
    fixed points of the quasipolynomialParams and gLVParams """
    print('TESTING fixed points are preserved in all algebraic forms')
    op = originalParams()
    qp = quasipolynomialParams(op)
    gp = gLVParams(qp)
    sp = ssrParams(gp)
    fps = op.get_stable_fps()
    qp_fps = qp.convert_op_to_qp(fps)
    gp_fps = gp.convert_qp_to_gp(qp_fps)
    sp_fps = sp.convert_gp_to_sp(gp_fps)
    verbose = True
    for op_fp,qp_fp,gp_fp,sp_fp in zip(fps, qp_fps, gp_fps, sp_fps):
        if verbose:
            print(op.get_integrand(op_fp, 0))
            print(qp.get_integrand(qp_fp, 0))
            print(gp.get_integrand(gp_fp, 0))
            print(sp.get_integrand(sp_fp, 0))
            print()

def verify_all_fps_are_stable():
    """ Test that the fixed points are stable in each system """
    print('TESTING fixed points are stable in all algebraic forms')
    op = originalParams()
    qp = quasipolynomialParams(op)
    gp = gLVParams(qp)
    sp = ssrParams(gp)

    for p in [op, qp, gp, sp]:
        print(p.fps)
        for fp in p.fps:
            print(np.linalg.eig(p.get_jacobian(fp))[0])
        print()

def verify_all_trajs_are_same():
    """ Plot the same trajectory in the three different forms (op, qp, gp) """
    print('TESTING all algebraic forms lead to same trajectory')
    op = originalParams()
    qp = quasipolynomialParams(op)
    print('ss are at u, v=')
    print(qp.ssa[0:2])
    print(qp.ssb[0:2])
    print(qp.ssa[0]*qp.ssb[1] - qp.ssa[1]*qp.ssb[0]*qp.gamma)

    gp = gLVParams(qp)
    sp = ssrParams(gp)
    fig, ax = plt.subplots(figsize=(6,6))
    op.plot_nullclines(ax, savefig=False)
    op.plot_example_traj(ax, savefig=False)
    qp.plot_example_traj(ax, savefig=False)
    gp.plot_example_traj(ax, savefig=False)
    sp.plot_example_traj(ax, savefig=True)


def sanity_checks():
    """ Run sanity check tests that ensure that algebraic manipulations don't
    change trajectories/fixed points/stability """
    #verify_all_fps_are_fps()
    #print('----')
    #verify_all_fps_are_stable()
    #print('----')
    verify_all_trajs_are_same()

if __name__ == '__main__':
    sanity_checks()

    #op = originalParams()
    #qp = quasipolynomialParams(op)
    #gp = gLVParams(qp)
    #sp = ssrParams(gp)


    #fig, ax = plt.subplots(figsize=(6,6))
    #x_null, y_null = op.get_nullclines()
    #true_fps = op.get_intersections(x_null, y_null)
    #print(true_fps)
