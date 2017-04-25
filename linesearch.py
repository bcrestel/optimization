"""
Line search algorithm
"""



def backtracking(f, x, alpha, direction, parameters_in=[]):
    """
    Implement backtracking line search algorithm

    Output:
        * status: True = success, False = line search failed
    """
    parameters = {}
    parameters['c1'] = 5e-5
    parameters['max_backtrack'] = 12
    parameters['rho'] = 0.5
    parameters.update(parameters_in)

    nb_backtrack = 0
    c1 = parameters['c1']
    assert(c1 > 0.0 and c1 < 1.0)
    rho = parameters['rho']
    assert(rho > 0.0 and rho < 1.0)
    assert(alpha > 0.0)

    xbk = x.copy()  # back-up copy
    cost_old = f.cost(xbk)
    grad = f.grad(xbk)
    gradxdirec = grad.dot(direction)
    assert(gradxdirec < 0.0)
    while nb_backtrack < parameters['max_backtrack']:
        x = xbk + alpha*direction
        cost_new = f.cost(x)
        if cost_new < cost_old + c1*alpha*gradxdirec:
            return True, alpha, x
        else:
            alpha *= rho

    return False, alpha, x
