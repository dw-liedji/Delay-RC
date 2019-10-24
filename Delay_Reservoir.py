#!/usr/bin/env python3

import numpy as np
import random

##########################################################################

class DelayReservoir():
    """
    Class to perform Reservoir Computing using Delay-Based Architecture
    """
    
    def __init__(self,N = 400,eta = 0.4,gamma = 0.05,theta = 0.2,loops=1,
            phi=np.pi/6):
        """
        Args:
            N:  Number of Virtual Nodes
            eta: Mackey Glass feedback strength
            gamma: Input Scaling
            theta: Distance between virtual nodes
            loops: Number of delay loops in reservoir
            phi: Phase (bias) for MZN
        """
        self.N = N
        self.eta = eta
        self.gamma = gamma
        self.theta = theta
        self.loops = loops
        self.phi = phi

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
        M_x = np.zeros((1+cycles,self.N*self.loops))
        J = self.mask(u,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N*self.loops)),J))

        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles+1):
            for j in range(self.loops-1,-1,-1):
                vn_0 = M_x[i-1,-1-self.N*j] + (-M_x[i-1,-1-self.N*j]\
                        +self.eta*(M_x[i-1,-1-self.N*j]+self.gamma*\
                        J[i-1,-1-self.N*j])/(1+M_x[i-1,-1-self.N*j]+\
                        self.gamma*J[i-1,-1-self.N*j]))*self.theta
                M_x[i,0+(self.loops-1-j)*self.N] = vn_0
            for j in range(1,self.N): 
                for k in range(self.loops):
                    vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                        self.eta*(M_x[i-1,j-1+self.N*k]+self.gamma* \
                        J[i-1,j-1+self.N*k])/(1+M_x[i-1,j-1+self.N*k]+\
                        self.gamma*J[i-1,j-1+self.N*k]))*self.theta
                    M_x[i,j+self.N*k] = vn

        
        #Remove first row of zeroes
        return M_x[1:]

    def calculateSerial(self,u,m):
        """
        Calculate reservoir state with serial loading, i.e. x(t)*J(t)

        Args:
            u: input data
            m: mask array

        Returns:
            M_x: Matrix of reservoir history 
        """

        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((1+cycles,self.N*self.loops))
        J = self.mask(u,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N*self.loops)),J))
        
        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles+1):
            for j in range(self.loops-1,-1,-1):
                vn_0 = M_x[i-1,-1-self.N*j] + (-M_x[i-1,-1-self.N*j]\
                        +1.0*(1e-12+self.gamma*M_x[i-1,-1-self.N*j]*\
                        J[i-1,-1-self.N*j])/(1+self.gamma*M_x[i-1,-1-self.N*j]*\
                        J[i-1,-1-self.N*j]))*self.theta*\
                        (self.loops-j)
                M_x[i,0+(self.loops-1-j)*self.N] = vn_0
            for j in range(1,self.N): 
                for k in range(self.loops):
                    vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                        1.0*(1e-12+self.gamma*M_x[i-1,j-1+self.N*k]* \
                        J[i-1,j-1+self.N*k])/(1+self.gamma*M_x[i-1,j-1+\
                        self.N*k]*J[i-1,j-1+self.N*k]))*self.theta
                    M_x[i,j+self.N*k] = vn
        
        #Remove first row of zeroes
        return M_x[1:]

    def calculateMZN(self,u,m):
        """
        Calculate rservoir state using Mach Zehnder activation function, i.e.
        sin^2(x+phi)

        Args:
            u: input data
            m: mask array

        Returns:
            M_x: matrix of reservoir history
        """

        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((1+cycles,self.N*self.loops))
        J = self.mask(u,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N*self.loops)),J))
        
        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles+1):
            for j in range(self.loops-1,-1,-1):
                vn_0 = M_x[i-1,-1-self.N*j] + (-M_x[i-1,-1-self.N*j]\
                        +self.eta*np.sin(M_x[i-1,-1-self.N*j]+self.gamma*\
                        J[i-1,-1-self.N*j]+self.phi)**2)*self.theta
                M_x[i,0+(self.loops-1-j)*self.N] = vn_0
            for j in range(1,self.N): 
                for k in range(self.loops):
                    vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                        self.eta*np.sin(M_x[i-1,j-1+self.N*k]+self.gamma* \
                        J[i-1,j-1+self.N*k]+self.phi)**2)*self.theta
                    M_x[i,j+self.N*k] = vn
        
        #Remove first row of zeroes
        return M_x[1:]

    def mutualCoupling(self,u,m):
        """
        Calculate reservoir with mutual coupling between (two) delay loops

        Args:
            u: input data
            m: mask array

        Returns:
            M_x: Matrix of reservoir history
        """

        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((1+cycles,self.N*2))
        J = self.mask(u,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N*2)),J))

        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles+1):
            for j in range(1,-1,-1):
                vn_0 = M_x[i-1,-1-self.N*j] + ( -M_x[i-1,-1-self.N*j]\
                        +self.eta*(M_x[i-1,-1-self.N*j]+\
                        0.05*M_x[i-1,-1-self.N*(1-j)]+self.gamma*\
                        J[i-1,-1-self.N*j])/(1+M_x[i-1,-1-self.N*j]+\
                        0.05*M_x[i-1,-1-self.N*(1-j)]+\
                        self.gamma*J[i-1,-1-self.N*j]))*self.theta*(2-j)
                M_x[i,0+(self.loops-1-j)*self.N] = vn_0
            for j in range(1,self.N): 
                for k in range(2):
                    vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                        self.eta*(M_x[i-1,j-1+self.N*k] \
                        + 0.05*M_x[i-1,j-1+self.N*(1-k)] + self.gamma* \
                        J[i-1,j-1+self.N*k])/(1+M_x[i-1,j-1+self.N*k]+\
                        0.05*M_x[i-1,j-1+self.N*(1-k)] +\
                        self.gamma*J[i-1,j-1+self.N*k]))*self.theta*(k+1)
                    M_x[i,j+self.N*k] = vn
        
        #Remove first row of zeros
        return M_x[1:]

    def calculateTwo(self,u,m,r = 1.0):
        """
        Calculates Reservoir State over the duration of u, two delay loops with
        different delay times but constant node spacing
        
        Args:
            u: Input data
            m: Mask array
            r: Ratio of t2 to t1

        Returns:
            M_x: Matrix of reservoir history
        """
        
        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((1+cycles,int(self.N*(1+r))))
        J = self.mask(u,m)
       
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,int(np.round(self.N*(1+r))))),J))

        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles+1):
            for j in range(1,-1,-1):
                vn_0 = M_x[i-1,-1-self.N*j] + (-M_x[i-1,-1-self.N*j]\
                        +self.eta*(M_x[i-1,-1-self.N*j]+self.gamma*\
                        J[i-1,-1-self.N*j])/(1+M_x[i-1,-1-self.N*j]+\
                        self.gamma*J[i-1,-1-self.N*j]))*self.theta
                M_x[i,0+(self.loops-1-j)*self.N] = vn_0
            for k in range(2): 
                if(k == 0):
                    for j in range(self.N):
                        vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                            self.eta*(M_x[i-1,j-1+self.N*k]+self.gamma* \
                            J[i-1,j-1+self.N*k])/(1+M_x[i-1,j-1+self.N*k]+\
                            self.gamma*J[i-1,j-1+self.N*k]))*self.theta
                        M_x[i,j+self.N*k] = vn
                else:
                    for j in range(int(self.N*r)):
                        vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                            self.eta*(M_x[i-1,j-1+self.N*k]+self.gamma* \
                            J[i-1,j-1+self.N*k])/(1+M_x[i-1,j-1+self.N*k]+\
                            self.gamma*J[i-1,j-1+self.N*k]))*self.theta
                        M_x[i,j+self.N*k] = vn


        
        #Remove first row of zeroes
        return M_x[1:,:400]


    def calculateTwoCont(self,u,m,r = 2.0):
        """
        Calculates Reservoir State over the duration of u, two delay loops with
        different delay times but constant node spacing, same mask length for
        both loops
        
        Args:
            u: Input data
            m: Mask array
            r: Ratio of t2 to t1

        Returns:
            M_x: Matrix of reservoir history
        """
        
        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        #Increase second loop length by factor of r
        M_x = np.zeros((1+cycles,self.N*(1+r)))
        J = self.mask(u,m)
       
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N*2)),J))

        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles+1):
            for j in range(1,-1,-1):
                vn_0 = M_x[i-1,-1-self.N*j] + (-M_x[i-1,-1-self.N*j]\
                        +self.eta*(M_x[i-1,-1-self.N*j]+self.gamma*\
                        J[i-1,-1-self.N*j])/(1+M_x[i-1,-1-self.N*j]+\
                        self.gamma*J[i-1,-1-self.N*j]))*self.theta
                M_x[i,0+(self.loops-1-j)*self.N] = vn_0
            for k in range(2): 
                if(k == 0):
                    for j in range(self.N):
                        vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                            self.eta*(M_x[i-1,j-1+self.N*k]+self.gamma* \
                            J[i-1,j-1+self.N*k])/(1+M_x[i-1,j-1+self.N*k]+\
                            self.gamma*J[i-1,j-1+self.N*k]))*self.theta
                        M_x[i,j+self.N*k] = vn
                else:
                    for j in range(self.N*2):
                        vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                            self.eta*(M_x[i-1,j-1+self.N*k]+self.gamma* \
                            J[i-1,(j-1)//r+self.N*k])/(1+M_x[i-1,j-1+self.N*k]+\
                            self.gamma*J[i-1,(j-1)//r+self.N*k]))*self.theta
                        M_x[i,j+self.N*k] = vn

        
        #Remove first row of zeroes
        return M_x[1:,:400]


