# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:32:02 2020

@author: siddarthaThentu
"""
from extract_features import extractFeatures
from sigmoid import sigmoid
import numpy as np

def predictTweets(tweet,freqs,theta):
    
    x = extractFeatures(tweet,freqs)
    
    y_pred = sigmoid(np.dot(x,theta))
    
    return y_pred