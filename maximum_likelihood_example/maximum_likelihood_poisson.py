# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:00:50 2016

@author: ricardo
"""

########
## Maximul likelihood estimator for the parameters alpha and beta of a
## poisson random variable using N independent observations
########

import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt


# Define the real parameters to generate the observations
alphareal = 100
betareal = 6

# Number of observations
N = 5

# Define vector of discrete time instants
t = np.arange(0,N)

# Define the (true) average value for the poisson RV in function of alpha and
# beta, which changes with each value of t
Mu = alphareal * np.exp(-betareal*t/6);

# Number of realizations to evaluate the average estimator's performance
runs = 1000

# Initialize variables to store the computed estimatives for each realization
alphahat = np.zeros((runs,1))
betahat = np.zeros((runs,1))


for k in range(0,runs):
    
    # Generate the observations as poisson random variables with the 
    # pre-defined values for the average value (lambda) for each "n"
    X = np.zeros((1,N));
    for n in range(1,N+1):
        X[0][n-1] = np.random.poisson(Mu[n-1])

    
    # Initialization for the optimization algorithm
    alphaest0 = 8 # 80
    betaest0 = 0.5 # 5
    # Concatenate in an array
    result0 = np.array([alphaest0, betaest0])
    
    # Define the objective function to be minimized:
    # Cost function of the MLE will be of the form:
    # cost(alpha,beta) = -sum_{t\in[0,N-1]}( -alpha * exp(-beta*t/6) 
    #                                + x[t]*(ln(alpha)-beta*t/6) - ln(x[t]!)),
    # where the last term is not necessary since it is constant in (alpha,beta)
    
    # Cost function input: params[0] => alpha, params[1] => beta
    fct = lambda params : np.sum( params[0] * np.exp(-params[1]*t/6) - X * (np.log(params[0])-params[1]*t/6) )
    result = sp_opt.minimize(fct, result0)
    # result = sp_opt.minimize(fct, result0, method='Powell')

    alphahat[k][0] = result.x[0]
    betahat[k][0] = result.x[1]
    
    
    
########
## Results
########

# Compute the mean value of the estimator
meanalpha = np.mean(alphahat)
meanbeta = np.mean(betahat)

# Compute the variance of the estimator
varalpha = np.var(alphahat)
varbeta = np.var(betahat)



print ("")
print ("Variance of alpha and beta estimation along MC runs:")
print (varalpha)
print (varbeta)

plt.hist(alphahat,bins=50)
plt.title('Histogram for the MLE of alpha')
plt.xlabel('MLE of alpha')
plt.show()

plt.hist(betahat,bins=50)
plt.title('Histogram for the MLE of beta')
plt.xlabel('MLE of beta')
plt.show()


    
    