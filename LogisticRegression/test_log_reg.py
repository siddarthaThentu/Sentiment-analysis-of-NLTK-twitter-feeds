# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:44:37 2020

@author: siddarthaThentu
"""
from predict_tweets import predictTweets
import numpy as np

def test_logistic_regression(test_x,test_y,freqs,theta):
    
    y_hat = []
    
    for tweet in test_x:
        y_pred = predictTweets(tweet,freqs,theta)
        if(y_pred>0.5):
            y_hat.append(1)
        else:
            y_hat.append(0)
            
    a=np.asarray(np.squeeze(y_hat))
    b=np.squeeze(test_y)
    result=(a==b)
    accuracy= result.mean()
    
    return accuracy
