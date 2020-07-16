# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:06:54 2020

@author: siddarthaThentu
"""
from sigmoid import sigmoid
import numpy as np

def gradientDescent(x,y,theta,alpha,num_iters):
    
    m = len(x)
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        mid = np.dot(np.transpose(y),np.log(h)) + np.dot(np.transpose(1-y),np.log(1-h))
        
        J = (-1.0/m)*mid
        
        # update the weights theta
        theta = theta - (alpha/m)*np.dot(np.transpose(x),h-y)
        
    ### END CODE HERE ###
    J = float(J)
    return J, theta