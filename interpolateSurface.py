# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 20:24:03 2017

@author: Yao Dong Yu

Description: fit data to option price surface against strike and
                time to maturity.
             Method is based on Ait-Sahalia and Duarte (2003)

             Assume no discount for now

Require: outputs of constrainedTransform.py: transform_main()
             transformed/ms_merged_call_*.csv
             transformed/ms_call_*.csv
             transformed/ms_put_*.csv
"""

import os
from os.path import isfile, join
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import loadDataOfDate as priceMapLoader
# Import R function
import rpy2.robjects as ro
ro.r('source("./smooth_K_impl.R")')

# Global plot settings
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.borderaxespad"] = 0

transformed_in_path = './data/transformed/'

poly_in_path = './data/local_poly/'
poly_out_path = './data/surface_map/'

def localPolySmoother_main(option_type='call'):
    files = os.listdir(transformed_in_path)
    files = [_f for _f in files if isfile(join(transformed_in_path, _f)) \
                                 and option_type in _f]
    files_path = [join(transformed_in_path, _f) for _f in files]

    # Set up R variables and run smoother
    ro.globalenv['K_step'] = 1
    ro.globalenv['K_order'] = 4
    for _f, _f_path in zip(files, files_path):
        ro.globalenv['in_path'] = _f_path
        ro.globalenv['out_path'] = poly_in_path + _f

        # Slove for local smoothed high order polynomial
        ro.r('smooth_K_impl(in_path, out_path, K_step, K_order)')

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def formatPoly(df_poly):
    all_Ks = df_poly['x'].unique()
    all_Ks.sort()

    price_poly = dict()
    for _t, _df in df_poly.groupby(['t']):
        _df_temp = pd.DataFrame(index=all_Ks, columns=['Price', 'V', 'Vk', 'Vkk'])
        #_df_temp.fillna(0, inplace=True)
        #price_poly[_t]
        _df_temp.ix[_df['x'], :] = _df[['beta0', 'beta2', 'beta3', 'beta4']].values
        _df_temp = _df_temp.astype(float)
        for _each in ['Price', 'V', 'Vk', 'Vkk']:
            _y = _df_temp[_each]
            nans_idx, func_interp = nan_helper(_y)
            _df_temp.ix[nans_idx, _each] = np.interp(func_interp(nans_idx),
                                                     func_interp(~nans_idx),
                                                     _y[~nans_idx])

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

def obtainSurface_impl(d_poly, Ks, ts):
    if ts is None:
        ts = np.array(list(d_poly.keys()))
        ts.sort()
    else:
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

    Price = np.array([[np.nan] * len(Ks)] * len(ts))
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
            elif curr_i == len(all_ts)-1:
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
        extrap_l_nan = np.array([np.nan] * len(Ks[Ks < min(_x)]))
        extrap_r = np.array([0] * len(Ks[Ks > max(_x)]))
        extrap_r_nan = np.array([np.nan] * len(Ks[Ks > max(_x)]))
        interp_x = Ks[np.all([Ks >= min(_x), Ks <= max(_x)], axis=0)]

        # create interpolation function, between points, linear is ok
        _y_Vt = np.array((_poly1['V'] - _poly0['V']) / ((_t1 - _t0) / 365))
        interp_Vt = interp1d(_x, _y_Vt)
        Vt[_i, :] = np.concatenate((extrap_l, interp_Vt(interp_x), extrap_r))

        _y_Price = np.array(_poly['Price'])
        interp_Price = interp1d(_x, _y_Price)
        Price[_i, :] = np.concatenate((extrap_l_nan, interp_Price(interp_x), extrap_r_nan))

        _y_V = np.array(_poly['V'])
        interp_V = interp1d(_x, _y_V)
        V[_i, :] = np.concatenate((extrap_l, interp_V(interp_x), extrap_r))

        _y_Vk = np.array(_poly['Vk'])
        interp_Vk = interp1d(_x, _y_Vk)
        Vk[_i, :] = np.concatenate((extrap_l, interp_Vk(interp_x), extrap_r))

        _y_Vkk = np.array(_poly['Vkk'])
        interp_Vkk = interp1d(_x, _y_Vkk)
        Vkk[_i, :] = np.concatenate((extrap_l, interp_Vkk(interp_x), extrap_r))

    # Reshape
    t_index_year = ts / 365
    df_Price = pd.DataFrame(Price, index=ts, columns=Ks)
    df_V = pd.DataFrame(V, index=t_index_year, columns=Ks)
    df_Vk = pd.DataFrame(Vk, index=t_index_year, columns=Ks)
    df_Vkk = pd.DataFrame(Vkk, index=t_index_year, columns=Ks)
    df_Vt = pd.DataFrame(Vt, index=t_index_year, columns=Ks)

    # Price do not need to be reduced
    df_V0 = reduceAllZeros(df_V)
    df_Vk0 = reduceAllZeros(df_Vk)
    df_Vkk0 = reduceAllZeros(df_Vkk)
    df_Vt0 = reduceAllZeros(df_Vt)

    # remove index of which all are 0
    t_inters = df_V0.index.union(df_Vk0.index).union(df_Vkk0.index).union(df_Vt0.index)
    K_inters = df_V0.columns.union(df_Vk0.columns).union(df_Vkk0.columns).union(df_Vt0.columns)

#    # keep square shape
#    # empirically, smaller t index and two tails of K have less data
#    min_size = min(len(t_inters), len(K_inters))
#    t_inters = t_inters[-min_size:]
#    K_n_extra = len(K_inters) - min_size
#    if K_n_extra is not 0:
#        K_i_left = K_n_extra - K_n_extra // 2
#        K_i_right = K_n_extra // 2
#        K_inters = K_inters[K_i_left:-(K_i_right+1)]

    # Reformat dataframes to common indexes
    df_V = df_V.ix[t_inters, K_inters]
    df_Vk = df_Vk.ix[t_inters, K_inters]
    df_Vkk = df_Vkk.ix[t_inters, K_inters]
    df_Vt = df_Vt.ix[t_inters, K_inters]

    return df_Price, df_V, df_Vk, df_Vkk, df_Vt, t_inters, K_inters

def obtainSurface(date, Ks, ts=None, option_type='call'):
    try:
        with open(poly_out_path + 'surface_poly_' + option_type + '_' + str(date) + '.pickle', 'rb') as fp:
            d_poly = pickle.load(fp)
    except FileNotFoundError:
        print("No available " + option_type + " option price for date " + str(date))
        return

    return obtainSurface_impl(d_poly, Ks, ts)

def reduceAllZeros(df):
    df = df.ix[:, ~(df == 0).all(axis=0)]
    df = df.ix[~(df == 0).all(axis=1), :]

    return df

def fitting_errors(option_type='call'):
    dates = priceMapLoader.getDates(option_type)
    err_metrics = pd.DataFrame(index=dates, columns=['N', 'MSE', 'RMSE', 'MAPE'])

    for _date in dates:
        df_real = priceMapLoader.loadDataOfDate(_date, option_type)
        df_fitted = obtainSurface(_date, df_real.columns, df_real.index, option_type)[0]
        # cast to interpolated area
        df_real = df_real.ix[df_fitted.index, df_fitted.columns]

        Ks = np.array(df_real.columns).astype(float)
        MSE = 0
        MAPE = 0
        count = 0
        for _t, _row in df_real.iterrows():
            K_idx = Ks[_row.notnull()]
            err = df_fitted.ix[_t, K_idx] - _row[K_idx]
            count += err.notnull().sum()
            err = err[err.notnull()]
            MSE = MSE + np.sum(np.square(err))
            MAPE = MAPE + np.sum(np.abs(err) / df_real.ix[_t, K_idx])

        err_metrics.ix[_date, 'MSE'] = MSE / count
        err_metrics.ix[_date, 'RMSE'] = np.sqrt(err_metrics.ix[_date, 'MSE'])
        err_metrics.ix[_date, 'MAPE'] = MAPE / count
        err_metrics.ix[_date, 'N'] = count

    err_metrics.to_csv('./data/error_metrics_' + option_type + '.csv')

    return err_metrics

def plot_errors(option_type='call'):
    df_err = pd.read_csv('./data/error_metrics_' + option_type + '.csv', index_col=0)
    dates = pd.to_datetime(np.array([str(_date) for _date in df_err.index]))

    fig, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(dates, df_err['MSE'])
    axarr[0].set_title('Mean Squared Error of interpolated surface')
    axarr[0].set_ylabel('MSE')
    axarr[1].plot(dates, df_err['RMSE'])
    axarr[1].set_title('Root Mean Squared Error of interpolated surface')
    axarr[1].set_ylabel('RMSE')
    axarr[2].plot(dates, df_err['MAPE'] * 100)
    axarr[2].set_title('Mean Absolute Percentage Error of interpolated surface')
    axarr[2].set_ylabel('MAPE (%)')

    fig.autofmt_xdate()
    plt.show()

def plotSurface(x, y, z, names, save=False, save_to="./_temp.png"):
    x_mesh, y_mesh = np.meshgrid(np.array(x).astype(float), np.array(y).astype(float))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x_mesh, y_mesh, z.T, cmap=cm.coolwarm)

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])

    plt.show()
