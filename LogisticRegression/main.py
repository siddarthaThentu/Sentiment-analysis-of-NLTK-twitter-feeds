# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:18:28 2020

@author: siddarthaThentu
"""
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import twitter_samples
from preprocess_tweets import preprocess_tweet
from gradient_descent import gradientDescent
from freqs import bld_freqs
from extract_features import extractFeatures
from test_log_reg import test_logistic_regression

#nltk.download("twitter_samples")
#nltk.download("stopwords")

all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

print(len(all_positive_tweets),len(all_negative_tweets))

#80% train and 20% test data split
test_pos = all_positive_tweets[4000:]
test_neg = all_negative_tweets[4000:]
train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

#create labels
train_y = np.append(np.ones((len(train_pos),1)),np.zeros((len(train_neg),1)),axis=0)
test_y = np.append(np.ones((len(test_pos),1)),np.zeros((len(test_neg),1)),axis=0)

print("train_shape = ",train_y.shape)
print("test_shape = ",test_y.shape)

print(train_x[0])
print(preprocess_tweet(train_x[0]))

freqDict,vocab = bld_freqs(train_x,train_y)

X = np.zeros((len(train_x),3))

for idx,tweet in enumerate(train_x):
    X[idx,:] = extractFeatures(tweet,freqDict)

Y = train_y   

print(X.shape)
print(Y.shape)


J, theta = gradientDescent(X,Y,np.zeros((3,1)),1e-9,1500)

accuracy = test_logistic_regression(test_x,test_y,freqDict,theta)

print("Accuracy of the model = ",accuracy)
    
    