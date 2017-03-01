# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:02:04 2017

@author: Yao Dong Yu

Description: pipeline for recovery from simulated option price data

Require: simulated in data in .csv format, with last least the following columns
    "strike_price", "days_to_mature", "option_price". Each .csv file should represent
    data of ONE option type (call/put) of ONE trading day.
"""

import numpy as np
import pandas as pd
import pickle
# Import R function
import rpy2.robjects as ro
ro.r('source("./smooth_K_impl.R")')

import reshapeData
import constrainedTransform as consTransform
import interpolateSurface
import continuous_recovery as recovery_cont

data_path = "./data/sim_pipline_output/"

def pipeline_reshape(filename, index_val):
    df_data = pd.read_csv(data_path + filename + ".csv")
    df_p_mx = reshapeData.constructPriceMatrix(df_data, reshapeData.fillPriceDf_sim)
    df_p_mx = reshapeData.cleanPriceMatrix(df_p_mx)

    df_p_mx.to_csv(data_path + filename +"_priceDf.csv")

    df_ms = consTransform.transformData(df_p_mx)
    df_ms.to_csv(data_path + filename + "_transformed_ms.csv")

    # Set up R variables
    ro.globalenv['in_path'] = data_path + filename + "_transformed_ms.csv"
    ro.globalenv['out_path'] = data_path + filename + "_poly_raw.csv"
    ro.globalenv['K_step'] = 1
    ro.globalenv['K_order'] = 4
    # Slove for local smoothed high order polynomial
    ro.r('smooth_K_impl(in_path, out_path, K_step, K_order)')

    # Reformat polynomial format into dictionary by time
    df_poly = pd.read_csv(data_path + filename + "_poly_raw.csv", index_col=0)
    dict_poly = interpolateSurface.formatPoly(df_poly)

    with open(data_path + filename + '_surface_poly.pickle', 'wb') as fp:
        pickle.dump(dict_poly, fp)

    ts = np.linspace(150, 500, 10)
    Ks = np.linspace(index_val - 500, index_val + 500, 50)

    V, Vy, Vyy, Vt, t_out, K_out = interpolateSurface.obtainSurface_impl(dict_poly, Ks, ts, 'call')
    V.to_csv(data_path + filename + "_V.csv")
    Vy.to_csv(data_path + filename + "_Vy.csv")
    Vyy.to_csv(data_path + filename + "_Vyy.csv")
    Vt.to_csv(data_path + filename + "_Vt.csv")

    try:
        recovered, m_hat, rho, S0, V_new, FoundPositive = \
            recovery_cont.RECOVERY(V, Vy, Vyy, Vt, index_val)
    except RuntimeError as err:
        print("Recovery Failed: ", err)
