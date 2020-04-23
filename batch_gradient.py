import numpy as np 
import pandas as pd  
print('Import libraries successfully')


# Setting parameter
etaList = [0.01, 0.1, 1, 10] # learning rate
n_iterations = 1000
m = 100
thetaList = [np.ones((2,1)), np.array([0,0]), np.array([-5,-5])]

def batchGradient(n_iterations, eta, theta, X_b, y): 
  for iteration in range(n_iterations):
     gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
     theta = theta - eta * gradients
  return theta

theta_best_list = {} 
for eta in etaList: 
  for theta in thetaList: 
    temp = batchGradient(n_iterations, eta, theta, X_b,y)
    theta_best_list.append(temp)
theta_best_list
print(batchGradient(n_iterations,eta, np.array([0,0]), X_b, y))