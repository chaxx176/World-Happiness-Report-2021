# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:15:58 2021

@author: K
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
 
from scipy.integrate import quad
from scipy.optimize import root

df=pd.read_csv('report_0809.csv')

def cal_dw_stat(res,lag):
    n = len(res)
    a = np.zeros(n)
    a[0] = -1
    a[lag] = 1
    a = np.expand_dims(a,axis=1)
    for i in range(n-lag-1):
        temp = np.zeros(n)
        temp[i+1] = -1
        temp[i+1+lag] = 1
        temp = np.expand_dims(temp, axis=1)
        a = np.concatenate([a,temp],axis=1)
 
    H = a.dot(a.T)
    dw_stat = res.dot(H.dot(res))/np.sum(np.square(res-np.mean(res)))
    return dw_stat

def test(n,k,lag,alpha=0.05,round_digit = 10):
    
    assert lag < n, 'invalid lag'
 
    ## make H matrix
    a = np.zeros(n)
    a[0] = -1
    a[lag] = 1
    a = np.expand_dims(a,axis=1)
    for i in range(n-lag-1):
        temp = np.zeros(n)
        temp[i+1] = -1
        temp[i+1+lag] = 1
        temp = np.expand_dims(temp, axis=1)
        a = np.concatenate([a,temp],axis=1)
 
    H = a.dot(a.T)
    ## get eigenvalue of H in increasing order
    eig_value, eig_vector = np.linalg.eig(H)
 
    idx = eig_value.argsort()
    eig_value = eig_value[idx]
 
    lower_eig_value = eig_value[1:n-k] ## eigen values for d L
    lower_eig_value = [round(x,round_digit) for x in lower_eig_value]
    upper_eig_value = eig_value[k+1:] ## eigen values for d U
    upper_eig_value = [round(x,round_digit) for x in upper_eig_value]
 
    def epsilon(u,mu,x):
        ## u scalar, mu vector
        return 0.5*np.sum(np.arctan(u*mu))
 
    def gamma(u,mu):
        ## u scalar, mu vector
        return np.power(np.prod(1+np.square(u*mu)),0.25)
    def F(x,eig_val):
        mu = eig_val-x
        f = lambda u: np.sin(epsilon(u,mu,x))/(u*gamma(u,mu))
        return 0.5-(1/np.pi)*quad(f,0,np.inf)[0]
 
    def objective(x,alpha,eig_val):
        return alpha-F(x,eig_val)
 
    lower_critical_val = root(objective,1,args=(alpha,lower_eig_value)).x[0]
    upper_critical_val = root(objective,2,args=(alpha,upper_eig_value)).x[0]
    
    return lower_critical_val, upper_critical_val


