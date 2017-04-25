"""
Solve unconstrained minimization problem min_x { f(x) }
using steepest descent, with backtracking line search
"""

import numpy as np
from optimization.linesearch import backtracking

def solve_SD(f, x0, PRINT=True, parameters_in=[]):
    """
    Input:
        * f must be a class with methods
            - cost
            - grad
        * x0: numpy array
        * parameters_in: dict for optional parameters

    Output:
        * True if optimization converged, otherwise False
    """
    parameters = {}
    parameters['max_iter'] = 20000
    parameters['abs_tol'] = 1e-12
    parameters['rel_tol'] = 1e-10


    x = x0.copy()
    iterSD = 0
    alpha = 1.0

    X = []
    GRAD = []

    if PRINT:   print '{:4} {:10} {:10}'.format('it', '|g|', 'alpha')
    while iterSD < parameters['max_iter']:
        grad = f.grad(x)
        normgrad = np.linalg.norm(grad)
        X.append(x.copy())
        GRAD.append(normgrad)
        if iterSD == 0:
            normgrad_init = normgrad
            if PRINT:   print '{:4} {:10} {:10}'.format(iterSD, normgrad, "")
        else:
            if PRINT:   print '{:4} {:10} {:10}'.format(iterSD, normgrad, alpha)

        if normgrad < parameters['abs_tol'] or \
        normgrad < parameters['rel_tol']*normgrad_init:
            if PRINT:   print 'Gradient sufficiently reduced'
            return True, X, GRAD

        status, alpha, x = backtracking(f, x, 2.*alpha, -1.0*grad, parameters)
        if not status:
            if PRINT:   print 'Line search failed'
            return False, X, GRAD

        iterSD += 1

    if PRINT:   print 'Maximum number of iterations reached'
    return False, X, GRAD
