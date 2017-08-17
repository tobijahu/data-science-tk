#!/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

help(np.zeros)

np.zeros( (3,4) )	# Create a matrix filled with zeros
'''
# Output will look like this:
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
'''


# Number series
np.arange( 10, 30, 5 )	# Create a series of numbers from
'''
# Output will look like this:
array([10, 15, 20, 25])
'''
np.arange( 0, 2, 0.3 )	# it accepts float arguments
'''
# Output will look like this:
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
'''


# pnorm, dnorm, qnorm, rnorm equivalents in python
from scipy.stats import norm
norm.ppf(.5)  # half of the mass is before zero. percentiles
norm.ppf(.5, loc=0, scale=1) # more paramters. percentiles
norm.cdf(0) # The inverse to ppf. Cumulative distribution function
norm.pdf(0) # Probability density function

from scipy.stats import chi2
chi2.ppf(.8, df=2) # two degress of freedom


# Plot a sine function with numpy and matplotlib
np.linspace( 0, 2, 9 )	# 9 numbers from 0 to 2
'''
# Output will look like this:
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
'''
from numpy import pi
x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
f = np.sin(x)
# do the plotting!
import matplotlib.pyplot as plt
plt.plot(x,f)
plt.ylabel('sin(x)')
plt.xlabel('x')
plt.grid()
plt.show()

# References:
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# Plot more complicated stuff
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

# References:
# https://matplotlib.org/users/pyplot_tutorial.html

# A scatter-plot
import matplotlib.pyplot as plt
x = np.arange(0, 1, 0.05)
y = np.power(x, 2)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.rc('grid', linestyle="-", color='black') # customized grid styling: solid line instead of dashed line
plt.scatter(x, y)
plt.grid()
plt.show()

# Plot to file e.g. .png
import numpy as np
import matplotlib as mpl
mpl.use('agg') # agg backend is used to create plot as a .png file
import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(9, 6)) # Create a figure instance
ax = fig.add_subplot(111) # Create an axes instance
bp = ax.boxplot(data_to_plot) # Create the boxplot
fig.savefig('fig1.png', bbox_inches='tight')

# References:
# https://stackoverflow.com/questions/8209568/how-do-i-draw-a-grid-onto-a-plot-in-python

# Matrices with numpy (The N-dimensional array)
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
type(x)
'''<type 'numpy.ndarray'>'''
x.shape
'''(2, 3)'''
x.dtype
'''dtype('int32')'''
x[1, 2] # The element of x in the *second* row, *third* column, namely, 6.


from numpy import array
mat = array(range(0,30)).reshape(3,10)
mat
'''
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
'''
## Return a single entry
mat[0,0]
'''
0
'''
## Return a column / slicing
mat[:, 0]
'''
array([ 0, 10, 20])

'''
## Return a submatrix
mat[:, 0:2]
'''
array([[ 0,  1],
       [10, 11],
       [20, 21]])
'''
mat[0:2,3]
'''
array([ 3, 13])
'''

# Compare two matrices
## Use == to check equality
from numpy import array
A = array(range(0,4)).reshape(2,2)
import copy
B = copy.copy(A)	# This is shallow copy. See also: copy.deepcopy(A)
(A==B).all()
B[1,0] = 9
(A==B).all()

np.array([1,1,1]) == np.array([1,1,0])

## Use array_equal() to check equality
np.array_equal([1, 2], [1, 2])
'''True'''
np.array_equal(np.array([1, 2]), np.array([1, 2]))
'''True'''
np.array_equal([1, 2], [1, 2, 3])
'''False'''
np.array_equal([1, 2], [1, 4])
'''False'''

# Count zeros in Matrix
np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]])
'''5'''
np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=0)
'''array([1, 1, 1, 1, 1])'''
np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)
'''array([2, 3])'''

# Obtain shape attribute of a matrix
np.array([1, 2, 3, 4]).shape
'''(4,)'''
np.zeros((2, 3, 4)).shape
'''(2, 3, 4)'''

# More Matrix stuff
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html


# Linear algebra
## Solving systems of equations
'''
Solve the system of equations
3 * x0 + x1 = 9 and x0 + 2 * x1 = 8
'''
a = np.array([[3,1], [1,2]])
b = np.array([9,8])
x = np.linalg.solve(a, b)
x
'''array([ 2.,  3.])'''
np.allclose(np.dot(a, x), b)	# Check the solution
'''True'''

# Math functions
import math
math.fmod(3.5,2) #modulo function for floats
'''1.5'''
3.5 % 2 #usual modulo function for ints
'''1.5'''
math.exp(0) #exponential function
'''1.0'''
math.log(1) #logarithm
'''0.0'''
math.pow(9.0, 2) #power (German: Exponentialfunktion)
'''81.0'''
math.sqrt(9) == math.pow(9.0, 1/2) #square root (German: Wurzel)
'''True'''

# Special functions
math.erf(1)
def phi(x):
	'Cumulative distribution function for the standard normal distribution'
	return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
phi(1)
'''0.841344746068543'''
from scipy.stats import norm
norm.cdf(1)
'''0.84134474606854293'''
math.gamma(1)
'''1.0'''

# Constants
math.pi
'''3.141592653589793'''
math.e
'''2.718281828459045'''
math.tau == 2 * math.pi
'''True'''

# more
math.isnan(math.nan)
'''True'''
math.exp(-math.inf)
'''0.0'''
import numpy as np
x = np.linspace( .1, .1, 10 )
x
'''array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])'''
sum(x)
'''0.99999999999999989'''
math.fsum(x)
'''1.0'''
math.isclose(sum(x), math.fsum(x), rel_tol=1e-09, abs_tol=0.0)
'''True'''
math.isclose(sum(x), math.fsum(x), rel_tol=1e-16, abs_tol=0.0)
'''False'''

# Runden, round, absolute
abs(-5)
'''5'''
math.ceil(x)
math.floor(x)

# Trigonometric functions
x = 0
[math.cos(x),math.sin(x),math.tan(x)]
'''[1.0, 0.0, 0.0]'''
def TrigTrip(number):
	return([math.cos(number),math.sin(number),math.tan(number)])

TrigTrip(math.pi/2)
'''[6.123233995736766e-17, 1.0, 1.633123935319537e+16]'''

# Resources:
# https://docs.python.org/3/library/math.html

from math import pi
import numpy as np
x = np.array([0,pi/2,pi,3*pi/2,2*pi])
np.sin(x)
'''
array([  0.00000000e+00,   1.00000000e+00,   1.22464680e-16,
        -1.00000000e+00,  -2.44929360e-16])
'''


# Pseudo-random numbers
import random
random.seed(a=0)
random.randint(0,1)
'''1'''
## Return a random choice
seq = (0,1,2,3)
random.choice(seq)
'''3'''

a = ()
random.shuffle

# Example of a function, while loop, break
def ExampleFunction(someInput):
	while someInput != 'test':
		print('Input: ' + someInput)
		break

ExampleFunction('1wwde')
'''
Input: 1wwde
'''

# While loop with user input
while True:
    n = input("Please enter 'hello':")
    if n.strip() == 'hello':
        break
'''
Please enter 'hello':
'''
#### Input converts all input to strings

# Example of a very simple function
def sum_two_numbers(a, b):
    return a + b


# Sampling
random.seed(a=0)
sample1 = np.random.normal(10, 20, 100) # normal distributed sample of size 100
sample2 = np.random.normal(0, 1, 20)

# Nullhypothesis: Sample distribution is normal
# Alternative: Sample is non-normal

# Choosing: Kolmogorov–Smirnov test
from scipy import stats
stats.kstest(sample1, 'norm')
'''KstestResult(statistic=0.68920220807031174, pvalue=0.0)'''
stats.kstest(sample2, 'norm')
'''KstestResult(statistic=0.17088982275315601, pvalue=0.55776000337770038)'''


# Bootstrapping / resampling examples
import numpy as np
from scipy import stats
sample1 = np.random.normal(10, 20, 1000) # sample from a normal distribution -- 100 values
np.random.choice(a=sample, size=(len(sample1),1000), replace=True) # bootstrap 1000 samples from sample
resultSample1 = 0
for i in range(0,1000):
	if(1 - stats.kstest(np.random.choice(a=sample1, size=len(sample1), replace=True), 'norm')[0] < 0.95):
		resultSample1 = resultSample1 + 1
print(resultSample1)
sample2 = np.random.normal(0, 1, 1000) # sample from a normal distribution -- 100 values
resultSample2 = 0
for i in range(0,1000):
	if(1 - stats.kstest(np.random.choice(a=sample2, size=len(sample2), replace=True), 'norm')[0] < 0.95):
		resultSample2 = resultSample2 + 1
print(resultSample2)


