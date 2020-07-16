# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:20:10 2020

@author: siddarthaThentu
"""
import numpy as np
from preprocess_tweets import preprocess_tweet

def bld_freqs(tweets,labels):
    
    labels_list = np.squeeze(labels).tolist()
    
    wordFreqs = {}
    
    for y,tweet in zip(labels_list,tweets):
        for word in preprocess_tweet(tweet):
            pair = (word,y)
            wordFreqs[pair] = wordFreqs.get(pair,0) + 1

    return wordFreqs        
        