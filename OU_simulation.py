# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:38:28 2017

@author: Siyuan Yao

Description: OU simulation function library
"""


import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy.matlib import repmat
from sklearn import datasets, linear_model

def OU_density(S0, theta, mu, sigma, K,T):
    # return P-measure density
    M=K.shape[0]
    N=T.shape[0]
    K2=repmat(K.reshape([1,-1]),N,1)
    T2=repmat(T.reshape([-1,1]),1,M)
    D=norm.pdf(K2, S0*np.exp(-theta*T2)+mu*(1-np.exp(-theta*T2)),np.sqrt(sigma**2/2/theta*(1-np.exp(-2*theta*T2))))
    return pd.DataFrame(D,index=T,columns=K)

def Simulated_AD(D,rho, m,S0):
    K=np.array(D.columns.tolist())
    T=np.array(D.index.tolist())
    M=K.shape[0]
    N=T.shape[0]
    K2=repmat(K.reshape([1,-1]),N,1)
    T2=repmat(T.reshape([-1,1]),1,M)
    return pd.DataFrame(np.exp(-rho*T2)*m(K2)/m(S0)*D.as_matrix(),index=D.index,columns=D.columns)

def Get_Four(K,T,eps_K,eps_T,S0, theta, mu, sigma,rho, m):
    V = Simulated_AD(OU_density(S0, theta, mu, sigma, K,T),rho, m,S0)
    Vt=1/eps_T *(Simulated_AD(OU_density(S0, theta, mu, sigma, K,T+0.5*eps_T),rho, m,S0).as_matrix()- \
                 Simulated_AD(OU_density(S0, theta, mu, sigma, K,T-0.5*eps_T),rho, m,S0).as_matrix())
    Vy=1/eps_K *(Simulated_AD(OU_density(S0, theta, mu, sigma, K+0.5*eps_K,T),rho, m,S0).as_matrix()- \
                 Simulated_AD(OU_density(S0, theta, mu, sigma, K-0.5*eps_K,T),rho, m,S0).as_matrix())
    Vyy=1/eps_K**2 *(Simulated_AD(OU_density(S0, theta, mu, sigma, K+eps_K,T),rho, m,S0).as_matrix()+ \
                     Simulated_AD(OU_density(S0, theta, mu, sigma, K-eps_K,T),rho, m,S0).as_matrix()- \
                     2*Simulated_AD(OU_density(S0, theta, mu, sigma, K,T),rho, m,S0).as_matrix())
    Vt=pd.DataFrame(Vt, index=T,columns=K)
    Vy=pd.DataFrame(Vy, index=T,columns=K)
    Vyy=pd.DataFrame(Vyy, index=T,columns=K)
    return V,Vt,Vy,Vyy


def Option_Generator(K,T,eta,S0, theta, mu, sigma,rho, m, style='C'):
    K2=np.arange(K[0],2*K[-1],0.1)
    D=OU_density(S0, theta, mu, sigma, K2,T)
    AD = Simulated_AD(D,rho, m,S0)
    Option = pd.DataFrame(index=T, columns=K)
    if style=='C':
        for t, row in Option.iterrows():
            flag=(np.random.random_sample(len(K))<eta)
            for j, k in enumerate(K):
                if flag[j]:
                    kk=K2[K2>=k]
                    row.iloc[j]=np.sum(AD.loc[t,k:].as_matrix().dot(kk-k)*0.1)
    else:
        for t, row in Option.iterrows():
            flag=(np.random.random_sample(len(K))<eta)
            for j, k in enumerate(K):
                if flag[j]:
                    kk=K2[K2<=k]
                    row[j]=np.sum(AD.loc[t,:k].as_matrix().dot(k-kk)*0.1)
    print('nb of observations:', len(K)*len(T)-Option.isnull().sum().sum())
    return Option


