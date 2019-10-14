#!/usr/bin/env python3

import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from Delay_Reservoir import DelayReservoir

############################################################################

def NARMA_Generator(length,u):
    """
    Generates NARMA10 Sequence

    Args:
        length: Length of series
        u: Input data

    Returns:
        y_k: NARMA10 series with k = length entries
    """

    #Generate first ten entries in series
    y_k = np.ones(10)*0.1

    #Iteratively calculate based on NARMA10 formula
    for k in range(10,length):
        t = 0.3*y_k[k-1]+0.05*y_k[k-1]*sum(y_k[k-1-i] for i in range(10))+1.5*u[            k-1]*u[k-10]+0.1
        y_k = np.append(y_k,t)

    return y_k


def NARMA_Test(test_length = 800,train_length = 800,plot = True):
    """
    Args:
        test_length: length of verification data
        train_length: length of training data

    Returns:
        NRMSE: Normalized Root Mean Square Error
    """
    
    #Randomly initialize u and m
    u = np.random.rand(train_length+test_length)/2.
    m = np.array([random.choice([-0.1,0.1]) for i in range(400)])
    
    #Calculate NARMA10 target
    target = NARMA_Generator(len(u),u)
    
    #Instantiate Reservoir, feed in training and verification datasets
    r1 = DelayReservoir()
    x = r1.calculate(u[:train_length],m)
    x_test = r1.calculate(u[train_length:],m)
    
    #Train using Ridge Regression
    clf = Ridge(alpha = 0)
    clf.fit(x,target[:train_length])
    y_test = clf.predict(x_test)
    
    #Calculate NRMSE
    NRMSE = np.sqrt(np.mean(np.square(y_test[50:]-target[train_length+50:]))            /np.var(target[train_length+50:]))
    
    #Plot predicted Time Series
    if(plot == True):
        plt.plot(y_test)
        plt.plot(target[train_length:])
        plt.title('NRMSE = %f'%NRMSE)
        plt.show()

    return NRMSE


print(NARMA_Test())
