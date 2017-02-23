# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 20:24:03 2017

@author: Yao Dong Yu

Description: fit data to option price surface against strike and
                time to maturity.
             Method is based on Ait-Sahalia and Duarte (2003)

             Assume no discount for now

Require: outputs of smooth_K.R:
             local_poly/ms_call_*.csv
             local_poly/ms_put_*.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Global plot settings
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.borderaxespad"] = 0

poly_in_path = './data/local_poly/'
poly_out_path = './data/surface_map/'

def formatPoly(df_poly):
    all_Ks = df_poly['x'].unique()
    all_Ks.sort()

    price_poly = dict()
    for _t, _df in df_poly.groupby(['t']):
        _df_temp = pd.DataFrame(index=all_Ks, columns=['V', 'Vk', 'Vkk'])
        _df_temp.fillna(0, inplace=True)
        #price_poly[_t]
        _df_temp.ix[_df['x'], :] = _df[['beta2', 'beta3', 'beta4']].values

        price_poly[_t] = _df_temp

    return price_poly

def formatPoly_main(option_type='call'):
    files = os.listdir(poly_in_path)
    files = [_f for _f in files if option_type in _f]

    for _f in files:
        _date_str = _f[-13:-4]
        df_poly = pd.read_csv(poly_in_path + _f)
        d_price_poly = formatPoly(df_poly)
        with open(poly_out_path + 'surface_poly_' + option_type + _date_str + '.pickle', 'wb') as fp:
            pickle.dump(d_price_poly, fp)

def obtainSurface(ts, Ks, date, option_type='call'):
    try:
        with open(poly_out_path + 'surface_poly_' + option_type + '_' + str(date) + '.pickle', 'rb') as fp:
            d_poly = pickle.load(fp)
    except FileNotFoundError:
        print("No available " + option_type + " option price for date " + str(date))
        return

    ts = np.array(ts)
    ts.sort()
    Ks = np.array(Ks)
    Ks.sort()

    all_ts = np.array(list(d_poly.keys()))
    all_ts.sort()

    t_min = min(all_ts)
    t_max = max(all_ts)
    if np.any(ts < t_min):
        print("t out of lower bound, cast to minimum t: ", t_min)
        ts = ts[ts >= t_min]

    if np.any(ts > t_max):
        print("t out of upper bound, cast to maximum t: ", t_max)
        ts = ts[ts <= t_max]

    V = np.array([[0.] * len(Ks)] * len(ts))
    Vk = np.array([[0.] * len(Ks)] * len(ts))
    Vkk = np.array([[0.] * len(Ks)] * len(ts))
    Vt = np.array([[0.] * len(Ks)] * len(ts))

    curr_i = 0
    for _i, _t in enumerate(ts):
        # all t's are sorted, find where current _t locates in array
        while _t > all_ts[curr_i]:
            curr_i += 1

        # Linear interpolate on t direction
        if _t == all_ts[curr_i]:
            if curr_i == 0:
                _t0 = _t
                _t1 = all_ts[curr_i+1]
            elif curr_i == len(all_ts):
                _t0 = all_ts[curr_i-1]
                _t1 = _t
            else:
                _t0 = all_ts[curr_i-1]
                _t1 = all_ts[curr_i+1]

            _poly0 = d_poly[_t0]
            _poly1 = d_poly[_t1]

            _poly = d_poly[_t]

        else:
            _t0 = all_ts[curr_i - 1]
            _t1 = all_ts[curr_i]
            _poly0 = d_poly[_t0]
            _poly1 = d_poly[_t1]

            _beta = (_t - _t0) / (_t1 - _t0)
            _poly = (1 - _beta) * _poly0 + _beta * _poly1

        _x = np.array(_poly.index)

        # all extrapolation should be 0 for V, Vk or Vkk
        extrap_l = np.array([0] * len(Ks[Ks < min(_x)]))
        extrap_r = np.array([0] * len(Ks[Ks > max(_x)]))
        interp_x = Ks[np.all([Ks >= min(_x), Ks <= max(_x)], axis=0)]

        # create interpolation function, between points, linear is ok
        _y_Vt = np.array((_poly1['V'] - _poly0['V']) / (_t1 - _t0))
        interp_Vt = interp1d(_x, _y_Vt)
        Vt[_i, :] = np.concatenate((extrap_l, interp_Vt(interp_x), extrap_r))

        _y_V = np.array(_poly['V'])
        interp_V = interp1d(_x, _y_V)
        V[_i, :] = np.concatenate((extrap_l, interp_V(interp_x), extrap_r))

        _y_Vk = np.array(_poly['Vk'])
        interp_Vk = interp1d(_x, _y_Vk)
        Vk[_i, :] = np.concatenate((extrap_l, interp_Vk(interp_x), extrap_r))

        _y_Vkk = np.array(_poly['Vkk'])
        interp_Vkk = interp1d(_x, _y_Vkk)
        Vkk[_i, :] = np.concatenate((extrap_l, interp_Vkk(interp_x), extrap_r))

    return V, Vk, Vkk, Vt, ts, Ks

def plotSurface(x, y, z, names):
    x_mesh, y_mesh = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm)
    except:
        ax.plot_surface(x_mesh, y_mesh, z.T, cmap=cm.coolwarm)

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    #ax.set_zlim(-0.00001, 0.00001)
    plt.show()
