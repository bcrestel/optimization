"""
Minimize Rosenbrock function
    f(x) = (a-x[0])**2 + b*(x[1]-x[0]**2)**2
"""

import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from optimization.steepest import solve_SD
from optimization.newton import solve_Newt
from optimization.bfgs import solve_bfgs

class Rosenbrock():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def cost(self, x):
        ax = self.a - x[0]
        yx = x[1] - x[0]*x[0]
        return ax*ax + self.b*yx*yx

    def grad(self, x):
        grad0 = -2.0*(self.a-x[0]) - 4.0*self.b*x[0]*(x[1]-x[0]*x[0])
        grad1 = 2.0*self.b*(x[1]-x[0]*x[0])
        return np.array([grad0, grad1])

    def hess(self, x):
        Hxx = 2.0 + 4.0*self.b*(3.0*x[0]*x[0]-x[1])
        Hxy = -4.0*self.b*x[0]
        Hyy = 2.0*self.b
        return np.array([[Hxx, Hxy], [Hxy, Hyy]])

    def plot(self, xmin, xmax, nx=100, ny=100):
        x = np.linspace(xmin[0], xmax[0], nx)
        y = np.linspace(xmin[1], xmax[1], ny)
        X, Y = np.meshgrid(x, y)
        F = self.cost(np.array([X.flatten(), Y.flatten()])).reshape((nx,ny))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot_surface(X, Y, F, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False, alpha=0.4)
        ax.view_init(60,-130)
        return fig, ax



if __name__ == "__main__":
    try:
        case = int(sys.argv[1])
    except:
        print 'Usage:\n\t{} <case nb>'.format(sys.argv[0])

    PLOT = False

    a = 1.0
    b = 100.
    f = Rosenbrock(a, b)

    x0 = np.zeros(2)
    x0[0] = -0.3
    x0[1] = 2.5

    if case == 0:
        print 'Check gradient against finite-difference'
        H = [1e-4, 1e-5, 1e-6]
        for ii in range(5):
            print 'point {}'.format(ii)
            x = np.random.randn(2)
            grad = f.grad(x)
            for jj in range(5):
                print '\tdirection{}'.format(jj)
                direction = np.random.randn(2)
                dirgrad = grad.dot(direction)
                for h in H:
                    cost1 = f.cost(x + h*direction)
                    cost2 = f.cost(x - h*direction)
                    gradFD = (cost1-cost2)/(2.0*h)
                    err = np.abs(dirgrad-gradFD)/np.abs(dirgrad)
                    print 'grad={}, gradFD={} (h={}), err={:.2e}'.format(
                    dirgrad, gradFD, h, err),
                    if err < 1e-8: 
                        print '\t==>> OK!!'
                        break
                    else:   print ''
        print '\nCheck Hessian against finite-difference'
        H = [1e-4, 1e-5, 1e-6]
        for ii in range(5):
            print 'point {}'.format(ii)
            x = np.random.randn(2)
            hess = f.hess(x)
            for jj in range(5):
                print '\tdirection{}'.format(jj)
                direction = np.random.randn(2)
                dirhess = hess.dot(direction)
                for h in H:
                    grad1 = f.grad(x + h*direction)
                    grad2 = f.grad(x - h*direction)
                    hessFD = (grad1-grad2)/(2.0*h)
                    err = np.linalg.norm(dirhess-hessFD)/np.linalg.norm(dirhess)
                    print 'hess={}, hessFD={} (h={}), err={:.2e}'.format(
                    dirhess, hessFD, h, err),
                    if err < 1e-8: 
                        print '\t==>> OK!!'
                        break
                    else:   print ''
    if case == 1:
        print 'Find minimizer using steepest descent'
        status, X, GRAD = solve_SD(f, x0)
        PLOT = True
        title = 'Steepest Descent'
    elif case == 2:
        print 'Find minimizer using Newton'
        status, X, GRAD = solve_Newt(f, x0, 
        parameters_in={'regularization':'identity'})
        PLOT = True
        title = 'Newton method'
    elif case == 3:
        print 'Find minimizer using BFGS'
        status, X, GRAD = solve_bfgs(f, x0)
        PLOT = True
        title = 'BFGS'
    # Plots
    if PLOT:
        X = np.array(X)
        # plot gradient convergence
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogy(GRAD)
        ax.set_title(title)
        ax.set_xlabel('iter')
        ax.set_ylabel('|g|')
        # plot path of minimization
        xmin = min(-1.5, X[:,0].min())
        xmax = max(1.5, X[:,0].max())
        ymin = min(-0.5, X[:,1].min())
        ymax = max(3.0, X[:,1].max())
        fig2, ax2 = f.plot(np.array([xmin,ymin]), np.array([xmax,ymax]))
        ax2.plot(X[:,0], X[:,1], 
        f.cost(np.array([X[:,0].flatten(), X[:,1].flatten()])),
        'k--o')
        ax2.scatter(
        np.array([X[0,0], X[-1,0]]), np.array([X[0,1], X[-1,1]]), 
        np.array([f.cost(np.array([X[0,0], X[0,1]])),
        f.cost(np.array([X[-1,0], X[-1,1]]))]),
        c='r', s=100)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Optimization path')
        # plot path of minimization (zoomed in)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(X[:,0], X[:,1])
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Optimization path -- (X,Y) projection')
        plt.show()

    if case == 5:
        print 'Compare all methods together'
        print '\nMethod 1: Steepest descent'
        status, X_SD, GRAD_SD = solve_SD(f, x0, PRINT=False)
        assert(status)
        X_SD = np.array(X_SD)
        GRAD_SD = np.array(GRAD_SD).reshape((len(GRAD_SD),-1))
        np.savetxt('SD.txt', np.concatenate((X_SD, GRAD_SD), axis=1),
        header='x y grad')

        print '\nMethod 2: Newton method with eigval truncation'
        status, X_Newt1, GRAD_Newt1 = solve_Newt(f, x0, PRINT=False, \
        parameters_in={'regularization':'eig'})
        assert(status)
        X_Newt1 = np.array(X_Newt1)
        GRAD_Newt1 = np.array(GRAD_Newt1).reshape((len(GRAD_Newt1),-1))
        np.savetxt('Newt1.txt', np.concatenate((X_Newt1, GRAD_Newt1), axis=1),
        header='x y grad')

        print '\nMethod 3: Newton method with identity truncation'
        status, X_Newt2, GRAD_Newt2 = solve_Newt(f, x0, PRINT=False, \
        parameters_in={'regularization':'identity'})
        assert(status)
        X_Newt2 = np.array(X_Newt2)
        GRAD_Newt2 = np.array(GRAD_Newt2).reshape((len(GRAD_Newt2),-1))
        np.savetxt('Newt2.txt', np.concatenate((X_Newt2, GRAD_Newt2), axis=1),
        header='x y grad')

        print '\nMethod 4: BFGS with damped update'
        status, X_bfgs, GRAD_bfgs = solve_bfgs(f, x0, PRINT=False)
        assert(status)
        X_bfgs = np.array(X_bfgs)
        GRAD_bfgs = np.array(GRAD_bfgs).reshape((len(GRAD_bfgs),-1))
        np.savetxt('BFGS.txt', np.concatenate((X_bfgs, GRAD_bfgs), axis=1),
        header='x y grad')
