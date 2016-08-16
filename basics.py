# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:58:03 2016

@author: bermudez
"""

# This file contains some useful operations in Phyton, usually employing NumPy

# Importing the modules necessary for scientific calculations
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import myfunctions as mf

# Creating an array of zeros - useful for initialization of vectors and matrices

mA = np.zeros( (3,4) )

# Creating an array of integer from 0 to N-1

N = 11
a = np.arange(N)

# a
#Out[8]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

# Creating the array as a bidimensional matrix 

N = 12
b = np.arange(N).reshape(3,N//3)    ##  // gives the floor division (integer part)

A=np.array(np.arange(12).reshape(3,4)) # 3x4 array constructed from a linear array

#A
#Out[4]: 
#array([[ 0,  1,  2,  3],
#       [ 4,  5,  6,  7],
#       [ 8,  9, 10, 11]])

# Same as b = np.arange(12).reshape(3,4)
#b
#Out[12]: 
#array([[ 0,  1,  2,  3],
#       [ 4,  5,  6,  7],
#       [ 8,  9, 10, 11]])

# How to plot a stem plot of a discrete-time signal
N = 20
h = mf.raiscos(N,0,2,2)
t = np.arange(N)
pl1 = plt.stem(t, h, linefmt='b-', markerfmt='bo', basefmt='k-'),plt.grid()
plt.tick_params(axis='both', labelsize=12)
plt.xlim(-0.5,20)
plt.ylim(-0.3,1.2)
plt.show()

# How to plot a vector using a continuous line
N = 200
h = mf.raiscos(N,0,2,2)
t = np.arange(N)
plt.plot(t,h),plt.grid()
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('time for $\sqrt{2}$',fontsize=14)
plt.ylabel('amplitude',fontsize = 14)
plt.ylim(-0.5,1.5)
plt.xlim(-10,210)
plt.show()

# How to plot more than one curve
N = 200
s1 = mf.raiscos(N,0,2,2)
s2 = mf.raiscos(N,0,1,0.5)
pl3 = plt.plot(t,s1,'b',t,s2,'r'),plt.grid()
plt.show()

# How to plot a histogram
X0 = np.sqrt((1-0.5**2)*1)*np.random.normal(0,1,size=10000)
plt4 = plt.hist(X0, bins=30, histtype='bar')  # raw histogram
plt.show()

pl5 = plt.hist(X0, bins=30, normed=1)  # normalized histogram (integral = 1)
plt.show()

# Generating random numbers

# Array of uniformly distributed discrete integer random numbers

#numpy.random.randint(low, high=None, size=None)
#This function returns random integers from 'low' (inclusive) 
#to 'high' (exclusive). In other words: randint returns random integers 
#from the "discrete uniform" distribution in the "half-open" interval 
#['low', 'high'). If 'high' is None or not given in the call, the results 
#will range from [0, 'low'). The parameter 'size' defines the shape of 
#the output. If 'size' is None, a single int will be the output. 
#Otherwise the result will be an array. The parameter 'size' defines 
#the shape of this array. So size should be a tuple. If size is defined 
#as an integer n, this is considered to be the tuple (n,).

# Examples

print(np.random.randint(1, 7))
#4
#
print(np.random.randint(1, 7, size=1))
#[5]
#
print(np.random.randint(1, 7, size=10))
#[4 5 6 3 4 3 2 3 6 4]
#
print(np.random.randint(1, 7, size=(5, 4)))
#[[4 1 6 3]
# [2 5 5 3]
# [5 3 4 2]
# [4 1 6 2]
# [6 1 1 4]]

# Sampling from a continuous uniform distribution in [0.,1.)
# Sampling an array of 7 elements
x = np.random.random_sample(7)
print(x)
#[ 0.76829484  0.31879245  0.40677395  0.23431267  0.87760548  0.32164243
#  0.78938758]

# sampling a matrix
M = np.random.random_sample((3, 4))
print(M) 
#[[ 0.53257061  0.06690095  0.06478968  0.92452651]
# [ 0.44056422  0.17956949  0.20518075  0.19147007]
# [ 0.8638584   0.12325188  0.69879374  0.73001331]]

# Uniform sampling from an arbitrary interval [a, b)
a = -3.4
b = 5.9
A = (b - a) * np.random.random_sample((3, 4)) + a
print(A)
#[[ 1.73289214  0.388571    5.15413665  3.54527801]
# [ 4.43251952 -2.55158719  5.0406861   2.2093405 ]
# [-0.55830661  2.87433891  5.09196123 -2.73186686]]

# Sampling from the standard normal distribution N(0.,1.)
print(np.random.randn(5,1))
#-0.17307073216246285

