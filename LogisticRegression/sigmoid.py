# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:02:54 2020

@author: siddarthaThentu
"""

import numpy as np

def sigmoid(z):
    
    return 1/(1+np.exp(-z))