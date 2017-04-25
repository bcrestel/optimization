"""
Solve unconstrained minimization problem min_x { f(x) }
using Newton's method regularized by eigenvalues truncation
"""

import numpy as np
from optimization.linesearch import backtracking

def solve_Newt(f, x0, PRINT=True, parameters_in=[]):
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
    parameters['cutoff_eig'] = 1e-8
    parameters['regularization'] = 'identity'
    parameters.update(parameters_in)

    x = x0.copy()
    iterNewt = 0

    X = []
    GRAD = []

    if PRINT:   print '{:4} {:10} {:10} {:10}'.format('it', '|g|', 'alpha', 'eig')
    while iterNewt < parameters['max_iter']:
        grad = f.grad(x)
        normgrad = np.linalg.norm(grad)
        X.append(x.copy())
        GRAD.append(normgrad)
        if iterNewt == 0:
            normgrad_init = normgrad
            if PRINT:   print '{:4} {:10.2e}'.format(iterNewt, normgrad)
        else:
            if PRINT:   
                print '{:4} {:10.2e} {:10.2e} {:10.2e} {:10.2e}'.format(
                iterNewt, normgrad, alpha, min(L), max(L))

        if normgrad < parameters['abs_tol'] or \
        normgrad < parameters['rel_tol']*normgrad_init:
            if PRINT:   print 'Gradient sufficiently reduced'
            return True, X, GRAD

        # compute search direction
        hess = f.hess(x)
        L, V = np.linalg.eigh(hess)

        if parameters['regularization'] == 'identity':
            # regularize by adding a multiple of the identity matrix
            Lr = parameters['cutoff_eig'] + L
        else:
            # regularize by truncating eigenvalues
            Lr = np.array([max(ll, parameters['cutoff_eig']) for ll in L])

        invL = np.diag(1./Lr)
        VTG = V.T.dot(grad)
        srchdir = - V.dot(invL.dot(VTG))

        status, alpha, x = backtracking(f, x, 1.0, srchdir, parameters)
        if not status:
            if PRINT:   print 'Line search failed'
            return False, X, GRAD

        iterNewt += 1

    if PRINT:   print 'Maximum number of iterations reached'
    return False, X, GRAD
