# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 02:17:15 2020

@author: siddarthaThentu
"""
import numpy as np

def train_naive_bayes(freqs,train_x,train_y):
    
    loglikelihood = {}
    
    #vocab
    vocab = set([pair[0] for pair in freqs])
    V = len(vocab)
    
    #Number of postive words and number of negative words
    N_pos = N_neg = 0
    
    for pair in freqs:
        if(pair[1]==1):
            N_pos+=1
        else:
            N_neg+=1
            
    D = len(train_y)
    D_pos = sum(np.array(train_y)==1)
    D_neg = D - D_pos
    
    logPrior = np.log(D_pos)-np.log(D_neg)
    
    for word in vocab:
        
        pos_freq = freqs.get((word,1),0)
        neg_freq = freqs.get((word,0),0)
        
        #probability and laplace smoothening
        prob_word_pos = (pos_freq+1)/(N_pos+V)
        prob_word_neg = (neg_freq+1)/(N_neg+V)
        
        loglikelihood[word] = np.log(prob_word_pos) - np.log(prob_word_neg)
        
    return loglikelihood,logPrior

    
            
    
    