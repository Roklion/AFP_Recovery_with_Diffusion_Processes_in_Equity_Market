# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:56:38 2017

@author: Yao Dong Yu

Description: fit data to option price surface against strike and
                time to maturity.
             Method is based on Ait-Sahalia and Duarte (2003)

             Assume no discount for now

Require: outputs of reshapeData.py:
             priceDfsMap_call.pickle
             priceDfsMap_put.pickle
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
# Load result of reshapeData.py
import loadDataOfDate as priceMapLoader

def transformObj(ms, ys):
    return np.square(ms - ys).sum()

def monoCons1(ms, xs, rt=0):
    return (ms[1] - ms[0]) / (xs[1] - xs[0]) + np.exp(-rt)

def monoCons2(ms):
    return ms[-2] - ms[-1]

def convexCons(ms, xs):
    return (ms[2] - ms[1]) / (xs[2] - xs[1]) - (ms[1] - ms[0]) / (xs[1] - xs[0])

def transformData(df_p_mx):
    # intermediate data m's
    df_ms = pd.DataFrame(index=df_p_mx.index, columns=df_p_mx.columns)

    # For each time to maturity, transform price against strike to
    #   convex and monotonically decreasing
    for _t, _itrow in df_p_mx.iterrows():
        _row_clean = _itrow.dropna(inplace=False)
        _prices = np.array(_row_clean.values.astype(float))
        _strikes = np.array(_row_clean.index.astype(float))
        _N = len(_strikes)

        # Need enough points to run optimization
        if _N < 10:
            continue

        # Initialize transformed values to be same as original option prices
        m0 = _prices
        # create list of constraints
        # first, simplified monotonically decreasing constraints
        cons = list()
        cons.append({'type':'ineq',
                     'fun' :lambda _ms: monoCons1(_ms, _strikes)})
        cons.append({'type':'ineq',
                     'fun' :lambda _ms: monoCons2(_ms)})
        # Then, append convexity constraints at each observation
        for _i in range(_N-2):
            cons.append({'type':'ineq',
                         'fun' :lambda _ms: convexCons(_ms[_i:_i+3], _strikes[_i:_i+3])})

        # All prices must be greater than 0
        bnds = ((0, None), ) * _N

        res = minimize(transformObj, m0, args=(_prices), method='SLSQP',
                       bounds=bnds, constraints=cons)

        df_ms.ix[_t, _row_clean.index] = res.x

    return df_ms

def transform_main(option_type='call'):
    dates = priceMapLoader.getDates(option_type)

    for _date in dates:
        df_ms = transformData(priceMapLoader.loadDataOfDate(_date, option_type))
        df_ms.to_csv('./data/transformed/ms_' + option_type + '_' + str(_date) + '.csv')