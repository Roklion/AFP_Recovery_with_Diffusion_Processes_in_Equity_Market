# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:10:22 2017

@author: kcfef
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Global plot settings
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.borderaxespad"] = 0

T_degree = 2
K_degree = 5

def polyK2ndCons(params, Xs_K_2nd, factor_K_2nd):
    return params * factor_K_2nd * Xs_K_2nd

def polyObjective(params, Xs, Y):
    return np.square((params * Xs).sum(axis=1) - Y).sum()

def constructPolynomial(df_p_mx):
    # extract coordinates of non-nan data
    rows, cols = np.where(df_p_mx.notnull())

    days_till_maturity = df_p_mx.index[rows]
    strikes = df_p_mx.columns[cols]
    Xs = list(zip(days_till_maturity, strikes))
    Y = [df_p_mx.ix[_row, _col] for _row, _col in Xs]

    # expand to desired degree with 2 variables
    poly = PolynomialFeatures(degree=T_degree+K_degree)
    expanded_Xs_all = poly.fit_transform(Xs)
    powers_all = poly.powers_

    # filter by desired degree on each axis
    idx_filter = np.where([_x[0] <= T_degree and _x[1] <= K_degree for _x in powers_all])
    powers = powers_all[idx_filter]
    expanded_Xs = np.array([_x[idx_filter] for _x in expanded_Xs_all])

    # Compute 2nd derivatrive (of K) polynomial factor and new exponent
    factor_K_2nd = np.array([_x[1]*(_x[1]-1) for _x in powers])

    power_K_2nd = np.array([[_x[0], np.maximum(0, _x[1]-2)] for _x in powers])
    power_K_2nd_idx = np.array([np.where(np.all(powers_all == _power, axis=1))[0][0]
                                for _power in power_K_2nd])
    Xs_K_2nd = np.array([_x[power_K_2nd_idx] for _x in expanded_Xs_all])

    return expanded_Xs, Y, factor_K_2nd, Xs_K_2nd, powers, Xs

def fitPolynomial(df_p_mx):
    # standardize variables
    df_p_mx.columns /= 400
    df_p_mx.index = np.sqrt(df_p_mx.index)

    Xs, Y, factor_K_2nd, Xs_K_2nd, powers, origX = constructPolynomial(df_p_mx)
    t_orig, K_orig = zip(*origX)

#    res = np.linalg.lstsq(Xs, Y)
#    res_params = res[0]
    # Solve minimization problem with 2nd order constraint (curvature w.r.t. K > 0)
    #params0 = (15.,) * Xs.shape[1]
    np.random.seed(1)
    params0 = np.random.rand(len(powers)) * 2000 - 1000
    params0 = (100, -10, 2200, 1.3, -75, -970, 0.55, 43, 154, -0.52, -7, -8, 0.099, 0.39, -0.0056, 1, 1, 1)
    res = minimize(polyObjective, params0, method='BFGS', args=(Xs, Y))

    res_params = res.x

    t = np.sqrt(np.linspace(0, 1000, 300))
    K = np.linspace(400, 3000, 300) / 400
    t_mesh, K_mesh = np.meshgrid(t, K)
    # Coefficients
    mx_c = np.matrix([[None] * (K_degree + 1)] * (T_degree + 1))
    for _idx, _pow in enumerate(powers):
        mx_c[_pow[0], _pow[1]] = res_params[_idx]
    # compute plane
    p_mesh = np.polynomial.polynomial.polyval2d(t_mesh, K_mesh, mx_c)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t_mesh, K_mesh, p_mesh, cmap=cm.coolwarm)
    ax.scatter(t_orig, K_orig, Y)
    plt.show()