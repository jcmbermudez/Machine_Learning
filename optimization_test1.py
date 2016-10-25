# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:06:40 2016

@author: ricardo
"""


import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt



print ("Compare a number of optimization methods when finding the minima of a cost function")

# Defien the variance of the noise to be added to the cost function
var_noise = 1

# Define the objective funciton to be minimized
func = lambda x: sp_opt.rosen(x) + np.sqrt(var_noise)*np.random.normal()
data_shape = (1,2);
# Bounds for the cost function:
func_bounds = None
# bounds=[(xmin,xmax)]



alg_choice = 3


# Define how wide is the range of points for the uniform random initialization
init_spread = 10

# Number of times the optimization is performed and averaged
n_runs = 10


"""
Overall view:
 - Powell             - Nonsmooth/noisy optimization
 - Nelder-Mead        - Nonsmooth/noisy optimization
 - Conjugate gradient - Cheap, not very fast convergence, few information needed
 - L-BFGS             - Estimate an approximation to the hessian. Between CG 
                        and Newton. Accepts bound constraints.
 - Newton             - Performs a full qudratic approximation of the function. 
                        Good if derivative and hessian information is 
                        available. Accepts bound constraints.
 - COBYLA             - Did not perform well. Poor convergence and sensible to 
                        initial conditions
 - SQLS               - Second order method, solves fairly general problems
                        with both equality and inequality constraints. SQLS 
                        can also use a lot of information about the problem, 
                        consisting of the derivative os the constraint functions.
"""


# Concatenate 
def sum_arrays_and_keep_shorter(a,b):
    a = np.array(a)
    b = np.array(b)
    if len(a) < len(b):
        c = b.copy()
        c[0:len(a)] += a
    else:
        c = a.copy()
        c[0:len(b)] += b
    return c

#------------------------------------------------------------------------------
if alg_choice == 1:
    print ("Using the Powell method! Solves 'min f(x)'  \n")
    print ("Works with non-smooth funtions, does not accepts derivative or hessian")
    # Does not use gradient, meaning it works well with noisy gradient (but is usually slower otherwise)
    
    cost_per_iter = np.array([])
    for i in range(0,n_runs):
        init = np.random.uniform(-init_spread,init_spread,data_shape)
        result = sp_opt.fmin_powell(func, init, ftol=0.0001, full_output=True, retall=True)
        cost_per_iter = sum_arrays_and_keep_shorter(cost_per_iter, [func(x) for x in result[6]])
        comp_cost = result[4] # number of function calls
    # Convert back to list
    cost_per_iter = (1/n_runs) * cost_per_iter
    cost_per_iter = cost_per_iter.tolist()
    
#       Interesting parameters:
#         ftol => Relative error in func(xopt) acceptable for convergence
        
        
#      Uses a modification of Powell’s method to find the minimum of a function 
#        of N variables. Powell’s method is a conjugate direction method.
#        The algorithm has two loops. The outer loop merely iterates over the 
#        inner loop. The inner loop minimizes over each current direction in 
#        the direction set. At the end of the inner loop, if certain conditions
#        are met, the direction that gave the largest decrease is dropped and
#        replaced with the difference between the current estimated x and the
#        estimated x from the beginning of the inner-loop.
#      The technical conditions for replacing the direction of greatest increase amount to checking that
#        no further gain can be made along the direction of greatest increase from that iteration.
#      The direction of greatest increase accounted for a large sufficient 
#        fraction of the decrease in the function value from that iteration of the inner loop.

#------------------------------------------------------------------------------
elif alg_choice == 2:
    print ("Using the Nelder-Mead Downhill Simplex method! Solves 'min f(x)'  \n")
    print ("Does not accepts derivative or hessian")
    # Does not use gradient, meaning it works well with noisy gradient (but is usually slower otherwise)
    
    cost_per_iter = np.array([])
    for i in range(0,n_runs):
        init = np.random.uniform(-init_spread,init_spread,data_shape)
        result = sp_opt.fmin_powell(func,init, ftol=0.0001, full_output=True, retall=True)
        cost_per_iter = sum_arrays_and_keep_shorter(cost_per_iter, [func(x) for x in result[6]])
        comp_cost = result[4] # number of function calls
    # Convert back to list
    cost_per_iter = (1/n_runs) * cost_per_iter
    cost_per_iter = cost_per_iter.tolist()
        
#       Interesting parameters:
#         ftol => Relative error in func(xopt) acceptable for convergence
        
        
#      Uses a Nelder-Mead simplex algorithm to find the minimum of function of one or more variables.
#      This algorithm has a long history of successful use in applications. 
#        But it will usually be slower than an algorithm that uses first or 
#        second derivative information. In practice it can have poor performance 
#        in high-dimensional problems and is not robust to minimizing complicated 
#        functions. Additionally, there currently is no complete theory 
#        describing when the algorithm will successfully converge to the minimum, 
#        or how fast it will if it does. Both the ftol and xtol criteria must 
#        be met for convergence.

#------------------------------------------------------------------------------
elif alg_choice == 3: #or n == 9 or n == 4:
    print ("Using the Nonlinear Conjugate Gradient method! Solves 'min f(x)'  \n")
    print ("Accepts derivative (gradient), does not accepts hessian")
    # Computationally cheaper than quasi-Newton algorithms although it uses more iterations. Better for cheap (simple) cost functions
    
    cost_per_iter = np.array([])
    for i in range(0,n_runs):
        init = np.random.uniform(-init_spread,init_spread,data_shape)
        result = sp_opt.fmin_cg(func,init, fprime=None, epsilon=1.4901161193847656e-08, gtol=1e-05, full_output=True, retall=True)
        cost_per_iter = sum_arrays_and_keep_shorter(cost_per_iter, [func(x) for x in result[5]])
        comp_cost = result[3] # number of function calls
    # Convert back to list
    cost_per_iter = (1/n_runs) * cost_per_iter
    cost_per_iter = cost_per_iter.tolist()
    
#       Interesting parameters:
#         fprime => A function that returns the gradient of f at x. If 'None' gradient is computed numerically
#         epsilon => Step size(s) to use when fprime is approximated numerically. Can be a scalar or a 1-D array. Defaults to sqrt(eps), with eps the floating point machine precision. Usually sqrt(eps) is about 1.5e-8.
#         gtol => Stop when the norm of the gradient is less than gtol.


#   This conjugate gradient algorithm is based on that of Polak and Ribiere [R156].
#   Conjugate gradient methods tend to work better when:
#     - f has a unique global minimizing point, and no local minima or other stationary points,
#     - f is, at least locally, reasonably well approximated by a quadratic function of the variables,
#     - f is continuous and has a continuous gradient,
#     - fprime is not too large, e.g., has a norm less than 1000,
#     - The initial guess, x0, is reasonably close to f ‘s global minimizing point, xopt.

#------------------------------------------------------------------------------
elif alg_choice == 4:
    print ("Using the Newton-CG method! Solves 'min f(x)'  \n")
    print ("== Unimplemented! ==")
    print ("Needs derivative (gradient), accepts Hessian")
    cost_per_iter = np.array([])
    
#------------------------------------------------------------------------------
elif alg_choice == 5:
    print ("Using the L-BFGS-B method! Solves 'min f(x)' with bound constraints!  \n")
    print ("Accepts derivative (gradient), computes inverse hessian implicitly (does not stores in memory)")
    # A low memory approximaiton of the BFGS algorithm (the full inverse hessian is not stored)
    # More costly per iteration than Conjugate Gradient, but with faster convergence
    
    cost_per_iter = np.array([])
    for i in range(0,n_runs):
        init = np.random.uniform(-init_spread,init_spread,data_shape)
        temp = []
        temp.append(init.reshape((2,)))
        def measure_convergence(xk):
            global temp
            temp.append(xk)
            #print(xk)
        result = sp_opt.fmin_l_bfgs_b(func,init, fprime=None, approx_grad=True, bounds=func_bounds, maxiter=300, m=10, factr=1e7, pgtol=1e-05, epsilon=1e-08, callback=measure_convergence, iprint=101, disp=101)
        cost_per_iter = sum_arrays_and_keep_shorter(cost_per_iter, [func(x) for x in temp])
        comp_cost = result[2]['funcalls'] # number of function calls
    # Convert back to list
    cost_per_iter = (1/n_runs) * cost_per_iter
    cost_per_iter = cost_per_iter.tolist()
    
#       Interesting parameters:
#         m => Maximum number of variable metric corrections used to define the limited memory matrix
#         approx_grad => True, except if the gradient is given by fprime or returned in fun
#         factr => Iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps, where eps is the machine precision, which is automatically generated by the code.
#                  Typical values for factr are: 1e12 for low accuracy; 1e7 for moderate accuracy; 10.0 for extremely high accuracy.
#         pgtol => The iteration will stop when max{|proj g_i | i = 1, ..., n} <= pgtol where pg_i is the i-th component of the projected gradient.
#         epsilon => Step size used when approx_grad is True, for numerically calculating the gradient


#       Optimize the function, f, whose gradient is given by fprime using the 
#       quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)

#------------------------------------------------------------------------------
elif alg_choice == 6:
    print ("Using the truncated Newton-CG method! Solves 'min f(x)' with bound constraints!  \n")
    print ("Accepts derivative (gradient) ")
    # A version of the Newton-CG algorithm that handles constrined problems    
    
    cost_per_iter = np.array([])
    for i in range(0,n_runs):
        init = np.random.uniform(-init_spread,init_spread,data_shape)
        temp = []
        temp.append(init.reshape((2,)))
        def measure_convergence(xk):
            temp.append(xk)
        result = sp_opt.fmin_tnc(func,init, fprime=None, approx_grad=True, bounds=func_bounds, epsilon=1e-08, maxCGit=-1, maxfun=None, pgtol=1e-05, callback=measure_convergence, disp=101)
#            (func, init, fprime=None, approx_grad=True, bounds=func_bounds, epsilon=1e-08, scale=None, offset=None, messages=15, maxCGit=-1, maxfun=None, eta=-1, stepmx=0, accuracy=0, fmin=0, ftol=-1, xtol=-1, pgtol=-1, rescale=-1, disp=None, callback=measure_convergence)
        cost_per_iter = sum_arrays_and_keep_shorter(cost_per_iter, [func(x) for x in temp])
        comp_cost = result[1] # number of function calls
    # Convert back to list
    cost_per_iter = (1/n_runs) * cost_per_iter
    cost_per_iter = cost_per_iter.tolist()

#       Interesting parameters:
#         epsilon => Used if approx_grad is True. The stepsize in a finite difference approximation for fprime
#         approx_grad => True, except if the gradient is given by fprime or returned in fun
#         maxCGit => Maximum number of hessian*vector evaluations per main iteration
#         maxfun => Maximum number of function evaluation
#         ....


#       The algorithm incoporates the bound constraints by determining the 
#       descent direction as in an unconstrained truncated Newton, but never 
#       taking a step-size large enough to leave the space of feasible x’s. 
#       The algorithm keeps track of a set of currently active constraints, 
#       and ignores them when computing the minimum allowable step size. 
#       (The x’s associated with the active constraint are kept fixed.) 

#------------------------------------------------------------------------------
elif alg_choice == 7:
    print ("Using the Constrained BY-Linear Approximation (COBYLA) method! Solves 'min f(x)' with 'g(x)>=0' constraints!  \n")
    print ("The constraints must be a sequence of functions")    
    print ("Does not accepts the derivative (gradient) or the hessian")
    # Based on linear approximations to the objective function and each constraint
    
    # Converts the bounds to functional constraints to agree with the example
    if func_bounds == None:
        lower = lambda x: 1
        upper = lambda x: 1
    else:
        lower = lambda x:  x-func_bounds[0][0]
        upper = lambda x: -x+func_bounds[0][1]
    func_cons = [lower, upper]
    
    cost_per_iter = np.array([])
    for i in range(0,n_runs):
        init = [1.5, 0.4]
#        init = np.random.uniform(-init_spread,init_spread,data_shape)
        result = sp_opt.fmin_cobyla(func, init, cons=func_cons, rhobeg=1, rhoend=1e-5, maxfun=500, catol=0.0002, disp=3)
        cost_per_iter = sum_arrays_and_keep_shorter(cost_per_iter, [func(x) for x in []])        
        comp_cost = [] # number of function calls
    # Convert back to list
    cost_per_iter = (1/n_runs) * cost_per_iter
    cost_per_iter = cost_per_iter.tolist()
    print ("Working badly, sensible to initial condition and RHO, takes too long to converge (and often does not)")
    
#       Interesting parameters:
#         rhobeg => Reasonable initial changes to the variables
#         rhoend => Final accuracy in the optimization
#                   (lower bound on the size of the trust region)
#         catol => Absolute tolerance for constraint violations

        
#          Suppose the function is being minimized over k variables. At the 
#        jth iteration the algorithm has k+1 points v_1,...,v_(k+1), an 
#        approximate solution x_j, and a radius RHO_j. (i.e. linear plus a 
#        constant) approximations to the objective function and constraint 
#        functions such that their function values agree with the linear 
#        approximation on the k+1 points v_1,.., v_(k+1). This gives a linear 
#        program to solve (where the linear approximations of the constraint 
#        functions are constrained to be non-negative).
#          However the linear approximations are likely only good approximations 
#        near the current simplex, so the linear program is given the further 
#        requirement that the solution, which will become x_(j+1), must be 
#        within RHO_j from x_j. RHO_j only decreases, never increases. The 
#        initial RHO_j is rhobeg and the final RHO_j is rhoend. In this way 
#        COBYLA’s iterations behave like a trust region algorithm.
#          Additionally, the linear program may be inconsistent, or the 
#        approximation may give poor improvement. For details about how these 
#        issues are resolved, as well as how the points v_i are updated, refer 
#        to the source code or the references below.
    print (result)

#------------------------------------------------------------------------------
elif alg_choice == 8:
    print ("Using the Sequential Least Squares method! Solves 'min f(x)' with 'h(x)=0' and 'g(x)>=0' constraints!  \n")
    print ("")    
    print ("Accepts the derivative (gradient) of the cost functions and of the constraints")
    #    
    
    # Converts the bounds to functional constraints to agree with the example
    if func_bounds == None:
        func_bounds = ()
           
    cost_per_iter = np.array([])
    for i in range(0,n_runs):
        init = np.random.uniform(-init_spread,init_spread,data_shape)
        temp = []
        temp.append(init.reshape((2,)))
        def measure_convergence(xk):
            temp.append(xk)
        result = sp_opt.fmin_slsqp(func, init, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=func_bounds, fprime=None, fprime_eqcons=None, fprime_ieqcons=None, iter=500, acc=1e-06, epsilon=1e-07, full_output=1, disp=3, callback=measure_convergence)
        # fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None, args=(), iter=100, acc=1e-06, iprint=1, disp=None, full_output=0, epsilon=1.4901161193847656e-08, callback=None)
        cost_per_iter = sum_arrays_and_keep_shorter(cost_per_iter, [func(x) for x in temp])        
        comp_cost = result[2] # number of function calls
    # Convert back to list
    cost_per_iter = (1/n_runs) * cost_per_iter
    cost_per_iter = cost_per_iter.tolist()
    
#    callback=measure_convergence
    

#       Interesting parameters:
#         eqcons(¢) => list of functions of length n such that 
#                      eqcons[j](x,*args) == 0.0 when successfully optimized
#         f_eqcons(¢) => Returns a 1-D array in which each element must equal 0.0
#                        when successfully optimized. If f_eqcons is specified, 
#                        eqcons is ignored.
#         ieqcons(£) => list of functions of length n such that 
#                       ieqcons[j](x,*args)>= 0.0 when successfully optimized
#         f_ieqcons(£) => Returns a 1-D ndarray in which each element must  
#                         be >=0.0 when successfully optimized. If f_ieqcons is 
#                         specified, ieqcons is ignored.
#         bounds => list of tuples specifying lower and upper bound for each 
#                   independent variable [(xl0, xu0),(xl1, xu1),...]. Infinite 
#                   values will be interpreted as large floating values.
#         fprime => function that evaluates the partial derivatives of func
#         fprime_eqcons => function of the form f(x, *args) that returns the 
#                          m by n array of equality constraint normals. If not 
#                          provided, the normals will be approximated. The 
#                          array returned by fprime_eqcons should be sized as 
#                          ( len(eqcons), len(x0) ).
#         fprime_ieqcons => function of the form f(x, *args) that returns the 
#                           m by n array of inequality constraint normals. If 
#                           not provided, the normals will be approximated. 
#                           The array returned by fprime_ieqcons should be 
#                           sized as ( len(ieqcons), len(x0) ).
#         epsilon => step size for finite-difference derivative estimates


#       The objective function and the constraints are twice continuously 
#         differentiable.
#       SQP methods solve a sequence of optimization subproblems, each of 
#         which optimizes a quadratic model of the objective subject to a 
#         linearization of the constraints. If the problem is unconstrained, 
#         then the method reduces to Newton's method. If the problem has only 
#         equality constraints, then the method is equivalent to applying 
#         Newton's method to the first-order optimality (KKT) conditions.
# source: https://en.wikipedia.org/wiki/Sequential_quadratic_programming

#------------------------------------------------------------------------------
elif alg_choice >= 9:
    print ("There is no such things as dragons!")






# Plots the convergence of the algorithm
plt.plot(10*np.log10(cost_per_iter))
plt.title('Convergence of the cost per iteration')
plt.ylabel('$10\log_{10}(f(x))$')
plt.xlabel('Iteration')
plt.show()





