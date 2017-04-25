"""
Solve unconstrained minimization problem min_x { f(x) }
using BFGS with damped update
"""

import numpy as np
from optimization.linesearch import backtracking




def apply_Hk(S, Y, R, d0, x_in):
    """
    Compute matvec between current Hk matrix (approx to inverse of Hessian)
    and a vector x_in:  
                        Hx_out = Hk * x_in
    The Hk matrix is the BFGS approximation to the inverse of the Hessian
    Computation done via double-loop algorithm
    """
    A = []
    Hx_out = x_in.copy()

    for s, y, r in zip(reversed(S), reversed(Y), reversed(R)):
        a = r * s.dot(Hx_out)
        A.append(a)
        Hx_out += -a*y

    Hx_out *= d0

    for s, y, r, a in zip(S, Y, R, reversed(A)):
        b = r * y.dot(Hx_out)
        Hx_out += (a - b)*s

    return Hx_out



def solve_bfgs(f, x0, PRINT=True, parameters_in=[]):
    """
    Input:
        * f must be a class with methods
            - cost
            - grad
            - hess
        * x0: numpy array
        * parameters_in: dict for optional parameters

    Output:
        * True if optimization converged, otherwise False
    """
    parameters = {}
    parameters['max_iter'] = 2000
    parameters['abs_tol'] = 1e-12
    parameters['rel_tol'] = 1e-10
    parameters['damp'] = 0.2
    parameters['d0_init'] = 1.0
    parameters.update(parameters_in)

    x = x0.copy()
    iterNewt = 0

    X = []
    GRAD = []
    grad = np.zeros(2)

    d0 = parameters['d0_init']
    damp = parameters['damp']
    S, Y, R = [], [], []

    if PRINT:   print '{:4} {:10} {:10} {:10}'.format('it', '|g|', 'alpha', 'theta')
    while iterNewt < parameters['max_iter']:
        grad_old = grad.copy()
        grad = f.grad(x)

        # Update BFGS
        if iterNewt > 0:
            s = alpha*srchdir
            y = grad - grad_old
            sy = s.dot(y)
            Hy = apply_Hk(S, Y, R, d0, y)
            yHy = y.dot(Hy)
            if sy < damp*yHy:
                theta = (1.0-damp)*yHy/(yHy-sy)
                s = theta*s + (1-theta)*Hy
                sy = s.dot(y)
            assert(sy > 0.0)
            S.append(s)
            Y.append(y)
            R.append(1./sy)
            if iterNewt == 1:
                d0 = sy/y.dot(y)

        normgrad = np.linalg.norm(grad)
        X.append(x.copy())
        GRAD.append(normgrad)
        if iterNewt == 0:
            normgrad_init = normgrad
            if PRINT:   print '{:4} {:10.2e}'.format(iterNewt, normgrad)
        else:
            if PRINT:   
                print '{:4} {:10.2e} {:10.2e} {:10.4f}'.format(
                iterNewt, normgrad, alpha, theta)

        if normgrad < parameters['abs_tol'] or \
        normgrad < parameters['rel_tol']*normgrad_init:
            if PRINT:   print 'Gradient sufficiently reduced'
            return True, X, GRAD

        srchdir = apply_Hk(S, Y, R, d0, -grad)

        status, alpha, x = backtracking(f, x, 1.0, srchdir, parameters)
        if not status:
            if PRINT:   print 'Line search failed'
            return False, X, GRAD

        iterNewt += 1

    if PRINT:   print 'Maximum number of iterations reached'
    return False, X, GRAD
