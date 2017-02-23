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

'''
t_step = 1
K_step = 5
K_order = 5
Gaussian_sigma = 50
'''

def transformObj(ms, ys):
    return np.square(ms - ys).sum()

def monoCons1(ms, xs, rt=0):
    return (ms[1] - ms[0]) / (xs[1] - xs[0]) + np.exp(-rt)

def monoCons2(ms):
    return ms[-2] - ms[-1]

def convexCons(ms, xs):
    return (ms[2] - ms[1]) / (xs[2] - xs[1]) - (ms[1] - ms[0]) / (xs[1] - xs[0])

'''
def Gaussian_smoother(X, Y, sigma, eps=0.01):
    Y_smooth = pd.Series(index=X)
    Y1 = pd.Series(index=X)
    Y2 = pd.Series(index=X)

    V = np.matrix(Y).T
    I = np.array(X).astype(float)
    for i in I:
        y = norm.pdf(i-I, 0, sigma).dot(V) / np.sum(norm.pdf(i-I, 0, sigma))
        Y_smooth[i] = y
        y1 = norm.pdf(i+eps-I, 0, sigma).dot(V) / np.sum(norm.pdf(i+eps-I, 0, sigma))
        Y1[i] = (y1 - y) / eps
        y2 = norm.pdf(i-eps-I, 0, sigma).dot(V) / np.sum(norm.pdf(i-eps-I, 0, sigma))
        Y2[i]= (y1 + y2 - 2*y) / eps**2

    map_D = lambda x: norm.pdf(x-I, 0, sigma).dot(V) / np.sum(norm.pdf(x-I, 0, sigma))

    return Y_smooth, Y1, Y2, map_D

def Linear_smoother(X, Y):
    model = nparam.KernelReg([Y], [X], 'c')
    mean, _ = model.fit()

    # This function should also extrapolate
    map_smooth = lambda x: model.fit(x)[0]
    map_D1 = lambda x: model.fit(x)[1]

    return mean, map_smooth, map_D1

def Linear_extrap_call(f_interp, X_left, Y_left):
    # Complete function with
    def func(_x):
        # only need to extrapolate left-end, because right-end already goes to flat
        if _x < X_left[0]:
            return Y_left[0] + (_x - X_left[0]) * (Y_left[1] - Y_left[0]) / (X_left[1] - X_left[0])
        else:
            return f_interp([_x])

    return func

def call_smoother(X, Y, X_step, l_bound, r_bound):
    _, interp_func, interp_func_d1 = Linear_smoother(X, Y)

    # linear extrapolate function out of range
    min_X = min(X)
    max_X = max(X)
    l = np.array([min_X, min_X+X_step])
    extrap_func = Linear_extrap_call(interp_func, l, interp_func(l))

    #X_c = np.arange(0, r_bound+X_step, X_step)
    X_c = np.append(np.append(np.arange(0, min_X, X_step), X),
                    np.arange(max_X+X_step, r_bound+X_step, X_step))
    Y_c = np.array([extrap_func(_x) for _x in X_c])

    # Smooth with Gaussian kernel again to obtain high-order differentiable function
    Y_G, YG1, YG2, Y_G_func = Gaussian_smoother(X_c, Y_c, Gaussian_sigma)

def fillCoefs(df_coefs):
    ts = np.array(df_coefs.index)
    range_coefs = df_coefs.columns

    df_interp_coefs = pd.DataFrame(index=np.arange(min(ts), max(ts)+t_step, t_step),
                                   columns=range_coefs)

    for _idx, _t in enumerate(ts[1:], 1):
        _interp_x = np.arange(ts[_idx-1], _t+t_step, t_step)
        _end_x = ts[[_idx-1, _idx]]
        _interp_coefs = [np.interp(_interp_x, _end_x, df_coefs.ix[_end_x, _j]) for _j in range_coefs]
        df_interp_coefs.ix[_interp_x, :] = np.transpose(_interp_coefs)

    return df_interp_coefs
'''

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