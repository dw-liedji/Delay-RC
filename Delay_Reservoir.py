#!/usr/bin/env python3

import numpy as np
import random

##########################################################################

class DelayReservoir():
    """
    Class to perform Reservoir Computing using Delay-Based Architecture
    """
    
    def __init__(self,N = 400,eta = 0.4,gamma = 0.05,tau = 80,theta = 0.2):
        """
        Args:
            N:  Number of Virtual Nodes
            eta: Mackey Glass feedback strength
            gamma: Input Scaling
            tau: Delay loop time
            theta: Distance between virtual nodes
        """
        self.N = N
        self.eta = eta
        self.gamma = gamma
        self.tau = tau
        self.theta = theta

    def mask(self,u,m = None):
        """
        Args:
            u: Input data
            m: Mask array

        Returns:
            J: Multiplexed (masked) data
        """
        if m.all() == None:
            m = np.array([random.choice([-0.1,0.1]) for i in range(N)])
        u = np.reshape(u,(-1,1))
        m = np.reshape(m,(1,-1))
        
        return u@m

    def calculate(self,u,m):
        """
        Calculates Reservoir State over the duration of u
        
        Args:
            u: Input data
            m: Mask array

        Returns:
            M_x: Matrix of reservoir history
        """
        
        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((1+cycles,self.N))
        J = self.mask(u,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N)),J))
        
        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles):
            vn_0 = M_x[i-1,-1] + (-M_x[i-1,-1]+self.eta*(M_x[i-1,-1]+
                self.gamma*J[i-1,0])/(1+M_x[i-1,0]+self.gamma*
                J[i-1,9]))*self.theta
            M_x[i,0] = vn_0
            for j in range(1,self.N): 
                vn = M_x[i,j-1] + (-M_x[i,j-1] + self.eta*(M_x[i-1,j-1]+
                    self.gamma*J[i-1,j-1])/(1+M_x[i-1,j-1]+self.gamma
                    *J[i-1,j-1]))*self.theta
                M_x[i,j] = vn
        
        #Remove first row of zeroes
        return M_x[1:]


