# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 01:17:15 2020

@author: siddarthaThentu
"""

from freqs import bld_freqs
from train_naiveB import train_naive_bayes
from predict_naiveB import predict_naive_bayes
from test_naiveB import test_naive_bayes
from preprocess_tweets import preprocess_tweet
import pdb
from nltk.corpus import stopwords,twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

probPosClass = len(all_positive_tweets)/(len(all_positive_tweets)+len(all_negative_tweets))
probNegClass = len(all_negative_tweets)/(len(all_positive_tweets)+len(all_negative_tweets))

logPrior = np.log(float(probPosClass)/probNegClass)

freq = bld_freqs(train_x,train_y)

loglikelihood,logPrior = train_naive_bayes(freq,train_x,train_y)

accuracy = test_naive_bayes(test_x,test_y,logPrior,loglikelihood)

print("Accuracy of the model = ",accuracy)