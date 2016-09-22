# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:05:47 2016

@author: ricardo
"""


########
## Maximum likelihood Estimator for the phase of a sinusoidal signal in WGN
########

import numpy as np
import scipy as sp
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt


## Define the parameters used in the simulation
# signal frequency
omega0 = 2 * np.pi * 0.08
# phase value to be estimated
phireal = np.pi / 4
# WGN variance
sigma2 = 0.05
# signal amplitude
A = 1
# number of observations used to compute each estimate
N = 20
# number of estimatives computed to build an histogram
runs = 5000

# Initialize the estimation with zeros
phihat = np.zeros((runs,1))



for k in range(0,runs):
    
    # Generate N observations in the X
    X = np.zeros((1,N))
    for n in range(1,N+1):
        # np.random.normal(loc=0.0, scale=1.0, size=None)
        X[0][n-1] = A*np.cos(omega0*(n-1)+phireal) + np.sqrt(sigma2)*np.random.normal()


    # Boundaries for the variables to be estimated (used during optimization)
    phimin = -np.pi
    phimax =  np.pi
    
    # Define the objective function to be minimized:
    # || X - A * cos(omega0*(n) + (phi) ||^2 , where (n) and (phi) are vectors 
    # containing the values for each time instants
    fct = lambda phi: np.sum(np.square(X-A*(np.cos(omega0*np.arange(0,N) + phi*np.ones((N,1)))) ))
    
    # Perform the optimization:
    result = sp_opt.minimize(fct, 0, bounds=[(phimin,phimax)] )
#    result = sp_opt.minimize(fct, 0, method='Powell')
    phihat[k][0] = result.x
    # Exhaustive search
#    result = sp_opt.brute(fct, (slice(phimin,phimax),), args=(), full_output=True, finish=sp.optimize.fmin)
#    phihat[k][0] = result[0]
    
########
### Results
########
    
# evaluation of the estimator's mean
meanphi = np.mean(phihat)

# evaluation of the variance of the estimator
Nvarphi = N*np.var(phihat)

print ("")
print ("Variance of the estimation along MC runs:")
print (Nvarphi)

plt.hist(phihat,bins=50)
plt.title('Histogram for MLE of Phi')
plt.xlabel('MLE of phi')
plt.show()







