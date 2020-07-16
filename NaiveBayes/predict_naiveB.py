# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 02:29:35 2020

@author: siddarthaThentu
"""
from preprocess_tweets import preprocess_tweet

def predict_naive_bayes(tweet,logprior,loglikelihood):
    
    p = 0
    
    for word in preprocess_tweet(tweet):
        if word in loglikelihood:
            p += loglikelihood[word]
            
    return p+logprior
            