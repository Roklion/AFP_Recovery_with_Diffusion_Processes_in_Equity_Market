# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:02:04 2017

@author: Yao Dong Yu

Description: pipeline for recovery from simulated option price data

Require: simulated data in .csv format, index being years to maturity,
            columns being strike value
"""

import numpy as np
import pandas as pd
import pickle
# Import R function
import rpy2.robjects as ro
ro.r('source("./smooth_K_impl.R")')

import constrainedTransform as consTransform
import interpolateSurface
import continuous_recovery as recovery_cont

data_path = "./data/sim_pipline_output/"

def pipeline_run(filename, index_val, delta=1000, bandwidth=400):
    df_data = pd.read_csv(data_path + filename + ".csv", index_col=0)
    df_p_mx = df_data.astype(float)

    # No need to clean price matrix for simulation data, as simulated price
    #  does not have price discreteness and maturity effect

    # Simulation always generate time in years
    df_p_mx.index = (df_data.index * 365).astype(int)

    df_p_mx.to_csv(data_path + filename +"_priceDf.csv")

    df_ms = consTransform.transformData(df_p_mx)
    df_ms.to_csv(data_path + filename + "_transformed_ms.csv")

    # Set up R variables
    ro.globalenv['in_path'] = data_path + filename + "_transformed_ms.csv"
    ro.globalenv['out_path'] = data_path + filename + "_poly_raw.csv"
    ro.globalenv['K_step'] = 1
    ro.globalenv['K_order'] = 4
    ro.globalenv['bw'] = bandwidth
    # Slove for local smoothed high order polynomial
    ro.r('smooth_K_impl(in_path, out_path, K_step, K_order, bw)')

    # Reformat polynomial format into dictionary by time
    df_poly = pd.read_csv(data_path + filename + "_poly_raw.csv")
    dict_poly = interpolateSurface.formatPoly(df_poly, sim=True)

    with open(data_path + filename + '_surface_poly.pickle', 'wb') as fp:
        pickle.dump(dict_poly, fp)

    ts = np.arange(0, 2.1, 0.1) * 365
    Ks = np.linspace(50, 2*index_val + 10, 100)
    Ks_Vy2 = np.linspace(50+0.1, 2*index_val + 10 + 0.1, 100)
    #Ks = np.linspace(index_val - delta, index_val + delta, 100)
    #Ks_Vy2 = np.linspace(index_val - delta + 0.1, index_val + delta + 0.1, 100)
    Price, V, Vy, Vyy, Vt, t_out, K_out = interpolateSurface.obtainSurface_impl(dict_poly, Ks, ts)
    _, _ , _Vy_p, _, _, _, _ = interpolateSurface.obtainSurface_impl(dict_poly, Ks_Vy2, ts)
    Vyy_fd= pd.DataFrame((_Vy_p.values - Vy.values) / 0.1, index=Vy.index, columns=Vy.columns)

    Price.to_csv(data_path + filename + "_Price.csv")
    V.to_csv(data_path + filename + "_V.csv")
    Vy.to_csv(data_path + filename + "_Vy.csv")
    Vyy.to_csv(data_path + filename + "_Vyy.csv")
    Vyy_fd.to_csv(data_path + filename + "_Vyy_fd.csv")
    Vt.to_csv(data_path + filename + "_Vt.csv")

    try:
        recovered, m_hat, rho, S0, V_new, FoundPositive = \
            recovery_cont.RECOVERY(V, Vy, Vyy_fd, Vt, index_val)
    except RuntimeError as err:
        print("Recovery Failed: ", err)
        return None, None, None

    return recovered, m_hat, rho

