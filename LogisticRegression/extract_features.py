# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:32:59 2020

@author: siddarthaThentu
"""
from preprocess_tweets import preprocess_tweet

def extractFeatures(tweet,dictionary):
    pos=0
    neg=0
    for word in preprocess_tweet(tweet):
        pos += dictionary.get((word,1),0)
        neg += dictionary.get((word,0),0)
    
    return [1,pos,neg]