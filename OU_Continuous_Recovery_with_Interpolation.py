# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 16:02:04 2017

@author: Yao Dong Yu

Description: script to run simulated recover through interpolation pipeline

Require: simulated in data in .csv format, with last least the following columns
    "strike_price", "days_to_mature", "option_price". Each .csv file should represent
    data of ONE option type (call/put) of ONE trading day.
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.matlib import repmat

import OU_simulation as OU_sim
import sim_pipeline as sim


data_path = "./data/sim_pipline_output/"

rho=0.1
def m(x):
    return (x/2000+1)**-3

theta=1
mu=2100
sigma=0.3*1950
S0=1950

K=np.arange(50.0,4000.0,5.)
T=np.arange(0.1,2.0,0.1)

for _i in np.arange(10, 30):
    np.random.seed(_i)

    for eta in [0.5, 0.3, 0.1]:
        print("eta="+str(eta)+", iteration="+str(_i))
        data=OU_sim.Option_Generator(K,T,eta,S0, theta, mu, sigma,rho, m, style='C')

        filename = "sim_test_S"+str(S0)+"_eta"+str(eta)+'_'+str(_i)
        data.to_csv(data_path + filename + ".csv")

        recovered, m_hat, _rho = sim.pipeline_run(filename, S0, delta=1500, bandwidth=300)
        if recovered is not None:
            print("rho=" + str(_rho))

            recovered.to_csv(data_path + filename + "_recovered.csv")
            recovered.T.plot()
            plt.show()

            m_hat.to_csv(data_path + filename + "_mhat.csv")
            m_hat.plot()
            plt.show()

            delta_K = recovered.columns[1] - recovered.columns[0]
            wts = repmat((recovered * delta_K).sum(axis=1), recovered.shape[1], 1).T
            adj_recovered = np.divide(recovered, wts)
            adj_recovered.to_csv(data_path + filename + "_adj_recovered.csv")
            adj_recovered.T.plot()
            plt.show()


# In[8]:

V,Vt,Vy,Vyy=OU_sim.Get_Four(K,T,0.01,0.001,S0, theta, mu, sigma,rho, m)

