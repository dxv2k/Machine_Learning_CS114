import numpy as np 
import matplotlib.pyplot as plt 
from sympy import *
print('Import libraries successfully')

# Gradient descent to find minimum of single-variable 
# derivative equation 
def gradient(learningRate, numIter, initX, theta):
  minX = initX
  theta = np.poly1d(theta) 
  for i in range(numIter): 
    derivTheta = theta.deriv() # Calculate derivation of equation
    minX = minX - learningRate*(derivTheta(minX))
  # draw graph while iterate through
  return minX

learningRate = 0.01 
numIter = 1000 
initX = 0 
theta = [1,4,3] 
print(gradient(learningRate,numIter, initX, theta))
