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
        different delay times but constant node spacing, both loops undergo
        same number of cycles
        
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

    def calculateDiffLoop(self,u,m,tau):
        """
        Calculates Reservoir State over the duration of u, length of delay loop
        different from length total node interval
        
        Args:
            u: Input data
            m: Mask array
            tau: Delay loop length

        Returns:
            M_x: Matrix of reservoir history
        """
        
        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((1+cycles)*self.N)
        J = self.mask(u,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N)),J))
        J = J.flatten(order = 'C')

        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,(cycles+1)*self.N): 
            vn = M_x[i-1] + (-M_x[i-1] +self.eta*(M_x[i-1-tau]+self.gamma* \
                J[i-1])/(1+M_x[i-1-tau]+self.gamma*J[i-1]))*self.theta
            M_x[i] = vn

        #Remove first row of zeroes
        M_x = M_x.reshape((cycles+1,self.N))
        return M_x[1:]

    def calculateMultiDiffLoop(self,u,m,tau):
        """
        Calculates Reservoir State over the duration of u, length of delay loop
        different from length total node interval, multiple delay loops that
        are running constantly (do not hold shorter delay lines constant)
        
        Args:
            u: Input data
            m: Mask array
            tau: Delay loop length

        Returns:
            M_x: Matrix of reservoir history
        """
        
        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((self.loops,(1+cycles)*self.N))
        J = self.mask(u,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N)),J))
        J = J.flatten(order = 'C')

        #Iteratively solve Mackey Glass Equation with Euler's Method
        for j in range(self.loops):    
            for i in range(1,(cycles+1)*self.N): 
                vn = M_x[j,i-1] + (-M_x[j,i-1] +self.eta*(M_x[j,i-1-tau]+\
                    self.gamma*J[i-1])/(1+M_x[j,i-1-tau]+self.gamma*J[i-1]))*\
                    self.theta
                M_x[j,i] = vn

        #Remove first row of zeroes
        M_x = M_x.reshape((cycles+1,self.N*2))

        return M_x[1:]

    def calculateMZNBit(self,u,m,bits):
        """
        Calculate rservoir state using Mach Zehnder activation function, i.e.
        sin^2(x+phi), with finite bit precision 

        Args:
            u: input data
            m: mask array
            bits: number of bit precision

        Returns:
            M_x: matrix of reservoir history
        """

        cycles = len(u)
        
        #Add extra layer to account for delay at t = 0
        M_x = np.zeros((1+cycles,self.N))
        u_new = np.zeros(cycles)
        for i in range(cycles):
            u_new[i] = DelayReservoir.ADC(u[i],0,0.5,bits)
        J = self.mask(u_new,m)
        
        #Add extra layer to match indexes with M_x
        J = np.vstack((np.zeros((1,self.N)),J))
        
        #Iteratively solve Mackey Glass Equation with Euler's Method
        for i in range(1,cycles+1):
            for j in range(self.loops-1,-1,-1):
                vn_0 = M_x[i-1,-1-self.N*j] + (-M_x[i-1,-1-self.N*j]\
                        +self.eta*np.sin(DelayReservoir.ADC(\
                        M_x[i-1,-1-self.N*j],0.16025,0.15825,\
                        bits)+self.gamma*\
                        J[i-1,-1-self.N*j]+self.phi)**2)*\
                        self.theta
                M_x[i,0+(self.loops-1-j)*self.N] = vn_0
            for j in range(1,self.N): 
                for k in range(self.loops):
                    if j < 400:
                        vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                            self.eta*np.sin(DelayReservoir.ADC(\
                            M_x[i-1,j-1+self.N*k],0.15825,0.16025,bits)+\
                            self.gamma*\
                            J[i-1,j-1+self.N*k]\
                            +self.phi)**2)*self.theta
                    else:
                        vn = M_x[i,j-1+self.N*k] + (-M_x[i,j-1+self.N*k] + \
                            self.eta*np.sin(\
                            M_x[i-1,j-1+self.N*k]+\
                            self.gamma*\
                            J[i-1,j-1+self.N*k]\
                            +self.phi)**2)*self.theta

                    M_x[i,j+self.N*k] = DelayReservoir.ADC(vn,0.15825,0.16025,\
                        bits)
        
        #Remove first row of zeroes
        return M_x[1:]

    def ADC(V,V_low,V_high,bits):
        """
        Convert analog voltage to digital with bits # of bits

        Args:
            V: number to be converted
            bits: bit precision
            V_high: high end of voltage range
            V_low: low end of voltage range

        Returns:
            b: bit representation of number
        """
        
        #Find bit that V is closest to
        V_tot = V_high - V_low
        Nb = int(((V-V_low)/V_tot)*2**bits)
        b = Nb*(V_tot/2**bits)+V_low
        
        return b
