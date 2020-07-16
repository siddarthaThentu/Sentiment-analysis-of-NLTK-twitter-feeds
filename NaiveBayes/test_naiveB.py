# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 02:45:02 2020

@author: siddarthaThentu
"""
import numpy as np
from predict_naiveB import predict_naive_bayes

def test_naive_bayes(test_x,test_y,logprior,loglikelihood):
    
    y_hat = []
    
    for tweet in test_x:
        if(predict_naive_bayes(tweet,logprior,loglikelihood)>0):
            y_hat.append(1)
        else:
            y_hat.append(0)
    
    a=np.asarray(np.squeeze(y_hat))
    b=np.squeeze(test_y)
    result=(a==b)
    accuracy= result.mean()
    
    return accuracy
    