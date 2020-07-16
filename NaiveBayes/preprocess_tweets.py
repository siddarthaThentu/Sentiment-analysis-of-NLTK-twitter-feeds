# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:17:15 2020

@author: siddarthaThentu
"""
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def preprocess_tweet(tweet_str):
    
    #remove text "RT"
    mod_tweet_str = re.sub(r'^RT[\s]+','',tweet_str)
    
    #remove hyperlinks
    mod_tweet_str = re.sub(r'https?:\/\/.*[\r\n]*','',mod_tweet_str)
    
    #remove hashtags
    mod_tweet_str = re.sub(r'#','',mod_tweet_str)
    
    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
    
    tweet_tokens = tokenizer.tokenize(mod_tweet_str)
    
    #import english stop words from NLTK library
    stopwords_english = stopwords.words("english")
    
    clean_tweet = []
    for word in tweet_tokens:
        if(word not in stopwords_english and word not in string.punctuation):
            clean_tweet.append(word)
    
    #stem words of the tweet
    stemmer = PorterStemmer()
    
    stemmed_tweet = []
    for word in clean_tweet:
        stem_word = stemmer.stem(word)
        stemmed_tweet.append(stem_word)
    
    return stemmed_tweet
    