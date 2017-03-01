
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:56:38 2017

@author: Siyuan Yao

Description: helper functions implementing the Recovery Theorem in countinous
                case
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
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
    Vt=1/eps_T *(Simulated_AD(OU_density(S0, theta, mu, sigma, K,T+0.5*eps_T),rho, m,S0).as_matrix()-                Simulated_AD(OU_density(S0, theta, mu, sigma, K,T-0.5*eps_T),rho, m,S0).as_matrix())
    Vy=1/eps_K *(Simulated_AD(OU_density(S0, theta, mu, sigma, K+0.5*eps_K,T),rho, m,S0).as_matrix()-                Simulated_AD(OU_density(S0, theta, mu, sigma, K-0.5*eps_K,T),rho, m,S0).as_matrix())
    Vyy=1/eps_K**2 *(Simulated_AD(OU_density(S0, theta, mu, sigma, K+eps_K,T),rho, m,S0).as_matrix()+                Simulated_AD(OU_density(S0, theta, mu, sigma, K-eps_K,T),rho, m,S0).as_matrix()-                    2*Simulated_AD(OU_density(S0, theta, mu, sigma, K,T),rho, m,S0).as_matrix())
    Vt=pd.DataFrame(Vt, index=T,columns=K)
    Vy=pd.DataFrame(Vy, index=T,columns=K)
    Vyy=pd.DataFrame(Vyy, index=T,columns=K)
    return V,Vt,Vy,Vyy

def Get_Discrete_AD(K,T,S0, theta, mu, sigma,rho, m):
    dK=0.0005*S0

    step=np.arange(dK,K[-1],dK)

    Discrete_AD=pd.DataFrame(index=T, columns=K)
    left=dK
    for k in K[:-1]:
        step=np.arange(left,k,dK)
        histo=Simulated_AD(OU_density(S0, theta, mu, sigma, step,T),rho, m,S0).as_matrix()
        Discrete_AD.loc[:,k]=(histo.sum(axis=1)-histo[:,0]/2-histo[:,-1]/2)*dK
        left=k

    # right bound
    step=np.arange(K[-2],K[-1]*2,dK)
    histo=Simulated_AD(OU_density(S0, theta, mu, sigma, step,T),rho, m,S0).as_matrix()
    Discrete_AD.loc[:,K[-1]]=(histo.sum(axis=1)-histo[:,0]/2-histo[:,-1]/2)*dK
    return Discrete_AD

def fit_PDE(V, Vy, Vyy, Vt, c=0):
    K=V.columns
    T=V.index
    D=pd.Series(index=K)
    a1=pd.Series(index=K)
    a0=pd.Series(index=K)
    conditional_number=pd.Series(index=K)
    for y in K.tolist():
        #get norm of vectors
        nyy=np.linalg.norm(Vyy.loc[:,y])
        ny=np.linalg.norm(Vy.loc[:,y])
        nv=np.linalg.norm(V.loc[:,y])

        Y=Vt.loc[:,y].as_matrix().reshape(-1,1)
        X=pd.concat([Vyy.loc[:,y]/nyy,Vy.loc[:,y]/ny,V.loc[:,y]/nv],axis=1).as_matrix()
        ev=np.linalg.eigvals(X.T.dot(X))
        conditional_number.loc[y]=max(ev)/min(ev)
        try:
            Beta=np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
            D[y]=Beta[0]/nyy
            a1[y]=Beta[1]/ny
            a0[y]=Beta[2]/nv
        except:
            D[y]=np.nan
            a1[y]=np.nan
            a0[y]=np.nan
    if c!=0:
        #print(conditional_number)
        tmp=conditional_number>conditional_number[conditional_number>0].min()*c
        D[tmp]=np.nan
        a1[tmp]=np.nan
        a0[tmp]=np.nan
    return D, a1,a0

def fit_PDE_constrained(V, Vy, Vyy, Vt,Dc):
    K=V.columns
    T=V.index
    D=pd.Series(index=K)
    a1=pd.Series(index=K)
    a0=pd.Series(index=K)
    for y in K.tolist():
        Y=Vt.loc[:,y].as_matrix().reshape(-1,1)
        X=pd.concat([Vyy.loc[:,y],Vy.loc[:,y],V.loc[:,y]],axis=1).as_matrix()
        D[y]=Dc

        bnds = ((None,None), (None,None))
        error=lambda x: np.linalg.norm(X[:,1:].dot(x.reshape(-1,1))+D[y]*X[:,0]-Y)
        res = minimize(error, [0,0], bounds=bnds, tol=1e-20)
        a1[y]=res.x[0]
        a0[y]=res.x[1]
    return D,a1,a0



def Gaussian_smoother(X, sigma,eps=0.001, combo=True):
    Y=pd.Series(index=X.index)
    Y1=pd.Series(index=X.index)
    Y2=pd.Series(index=X.index)

    V=X.dropna().as_matrix()
    I=X.dropna().index.tolist()
    I=np.array([float(i) for i in I])

    regr = linear_model.LinearRegression()
    regr.fit(I.reshape(-1,1), V.reshape(-1,1))
    R2=regr.score(I.reshape(-1,1), V.reshape(-1,1))
    if 1-combo:
        R2=0

    for i in Y.index.tolist():
        y=norm.pdf(float(i)-I,0,sigma).dot(V)/np.sum(norm.pdf(float(i)-I,0,sigma))
        Y[i]=(1-R2)*y+R2*regr.predict(i)
        y1=norm.pdf(float(i)+eps-I,0,sigma).dot(V)/np.sum(norm.pdf(float(i)+eps-I,0,sigma))
        Y1[i]=(1-R2)*(y1-y)/eps+R2*regr.coef_[0]
        y2=norm.pdf(float(i)-eps-I,0,sigma).dot(V)/np.sum(norm.pdf(float(i)-eps-I,0,sigma))
        Y2[i]=(1-R2)*(y1+y2-2*y)/eps**2

    map_D= lambda x:(1-R2)*norm.pdf(x-I,0,sigma).dot(V)/np.sum(norm.pdf(x-I,0,sigma))+R2*float(regr.predict(x))

    return Y, Y1,Y2, map_D

def Get_Smoothed_Kappa(Dy,a1,sigma, combo=True):
    kappa=2*Dy-a1
    #kappa=-a1
#     if min(kappa)<-1000:
#         kappa=kappa-min(kappa)
#     if max(kappa)>1000:
#         kappa=kappa/max(kappa)*1000

    return Gaussian_smoother(kappa, sigma)

def Get_Smoothed_r(Dyy,kappa1,a0,sigma, combo=True):
    r=Dyy-kappa1-a0
    #r=-kappa1-a0
#     if min(r)<0:
#         r=r-min(r)
#     if max(r)>0.1:
#         r=r/max(r)*0.1
    return Gaussian_smoother(r, sigma)

def C_Recovery(dx, rhomax, NoSteps, D,r,k):
    rhomin=0.01
    N=D.shape[0]
    zapp=np.zeros([N,1])
    FoundPositive=0
    for n in range(0,NoSteps):
        rho=(rhomax+rhomin)/2
        #print(n,rho)
        z=np.zeros([N,2])
        Mid=int(np.floor((N)/2))-1
        z[Mid-1:Mid+2,0]=[1,1,1]
        z[Mid-1:Mid+2,1]=[1-dx,1,1+dx]
        for j in range(Mid+1,N-1):
            vj=k[j]*dx/(2*D[j])
            z[j+1,:]=1/(1+vj)*((2-dx**2*(rho-r[j])/D[j])*z[j,:]-(1-vj)*z[j-1,:])
        for j in np.arange(Mid-1,0,-1):
            vj=k[j]*dx/(2*D[j])
            z[j-1,:]=1/(1-vj)*((2-dx**2*(rho-r[j])/D[j])*z[j,:]-(1+vj)*z[j+1,:])

        Roots=np.sum(z[1:,:]*z[:-1,:]<=0,axis=0)
        if Roots[0]>1 or Roots[1]>1 :
            rhomax=rho
        elif Roots[0]==0:
            rhomin=rho
            zapp=z[:,0]
            FoundPositive=1
        elif Roots[1]==0:
            rhomin=rho
            zapp=z[:,1]
        else:
            A1=np.angle(z[:,0]+z[:,1]*1j)
            A2=np.angle(-(z[:,0]+z[:,1]*1j))
            if max(A1)-min(A1)<np.pi:
                rhomin=rho
                A=0.5*(max(A1)+min(A1))
                zapp=np.cos(A)*z[:,0]+np.sin(A)*z[:,1]
                FoundPositive=1
            elif (max(A2)-min(A2)<np.pi):
                rhomin=rho
                A=0.5*(max(A2)+min(A2))+np.pi
                zapp=np.cos(A)*z[:,0]+np.sin(A)*z[:,1]
                FoundPositive=1
            else:
                rhomax=rho
    if FoundPositive==0:
        #print('Did not Find a positive kernel')
        m=0.5/z[:,0]+0.5/z[:,1]
        raise RuntimeError("Did not find a positive kernel")
    else:
        m=1/zapp
        print('Positive Kernel found!')

    return rho,m, FoundPositive


def get_f(t,mx,m_hat,rho, p):
    my=m_hat.iloc[m_hat.index.searchsorted(p.index.tolist())]
    my.index=p.index
    return p*np.exp(rho*t)*mx/my

def RECOVERY(V, Vy, Vyy, Vt,S0, con_thres=400):
    # change data type for the columns
    V.columns=[float(i) for i in V.columns.tolist()]
    Vy.columns=[float(i) for i in Vy.columns.tolist()]
    Vyy.columns=[float(i) for i in Vyy.columns.tolist()]
    Vt.columns=[float(i) for i in Vt.columns.tolist()]
    # fit diffustion parameters
    D,a1,a0= fit_PDE(V, Vy, Vyy, Vt, con_thres)
    # smoothen and get D, Dy, Dyy
    Smoothed_D,Smoothed_Dy,Smoothed_Dyy, map_D =Gaussian_smoother(D, 200)
    # get smoothened kappa
    kappa, kappa1,kappa2, map_kappa=Get_Smoothed_Kappa(Smoothed_Dy,a1,200,combo=True)
    kappa.index=a1.index
    # get smoothened r
    r, r1,r2, map_r=Get_Smoothed_r(Smoothed_Dyy,kappa1,a0,200, combo=True)

    # make plots
    #plt.plot(Smoothed_D)
    #plt.show()
    #plt.plot(kappa)
    #plt.show()
    #plt.plot(r)
    #plt.show()

    # generate fundamental ODE parameter grid
    dx=0.1
    output_grid=np.arange(50,4000,dx)
    final_D=np.array([map_D(i) for i in output_grid])
    final_r=np.array([map_r(i) for i in output_grid])
    final_k=np.array([map_kappa(i) for i in output_grid])
    # recover!
    rho,m,FoundPositive=C_Recovery(dx, 0.5, 30, final_D,final_r,final_k)
    # packing
    m_hat=pd.Series(m,index=output_grid)
    # get m(x) and recovered probability
    mx=m_hat.iloc[m_hat.index.searchsorted(S0)]
    recovered=pd.DataFrame(index=V.index, columns=V.columns)
    for t,row in recovered.iterrows():
        recovered.loc[t,:]=get_f(t,mx,m_hat,rho,V.loc[t,:])
    # get a lambda function to calculate recovered probability
    return recovered, m_hat, rho, S0, V, FoundPositive