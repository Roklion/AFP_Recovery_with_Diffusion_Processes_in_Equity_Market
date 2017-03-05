# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:00:33 2017

@author: Yao Dong Yu

Description: sumary statistics and plot of cleaned data

Require: outputs of reshapeData.py:
             priceDfsMap_call.pickle
             priceDfsMap_put.pickle
"""

import numpy as np
import pandas as pd
import pickle

import loadDataOfDate as priceMapLoader
import interpolateSurface as interpSurface

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Global plot settings
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["legend.borderaxespad"] = 0

def summarizeDataOfDate(priceDf):
    df_shape = priceDf.shape
    N_data = priceDf.notnull().sum().sum()

    return df_shape + (N_data, )

def summarizeDataStats(priceDfs):
    dates = list(priceDfs.keys())
    dates.sort()

    stats_col = ['N_maturity', 'N_strike', 'N_data']
    df_stats = pd.DataFrame(index=dates, columns=stats_col)
    for _date in dates:
        df_stats.ix[_date, :] = summarizeDataOfDate(priceDfs[_date])

    return df_stats

def summarizeDataStats_main():
    for _type in ['call', 'put']:
        with open('./data/priceDfsMap_' + _type + '.pickle', 'rb') as fp:
            date_priceDf_map = pickle.load(fp)

        stats = summarizeDataStats(date_priceDf_map)

        fig = plt.figure()
        x_dates = pd.to_datetime(stats.index, format='%Y%m%d')

        ax1 = fig.add_subplot(211)
        ax2 = ax1.twinx()
        k_line, = ax1.plot(x_dates, stats['N_strike'], 'b')
        t_line, = ax2.plot(x_dates, stats['N_maturity'], 'r')

        ax1.xaxis.grid(True)
        ax1.set_title('Number of unique strike prices and maturity time',
                      fontsize=15)
        ax1.legend([k_line, t_line], ['Strike', 'Maturity Date'], loc=0)
        ax1.set_ylabel('Number of Strike Prices')
        ax2.set_ylabel('Number of Maturity Dates')

        ax3 = fig.add_subplot(212)
        ax4 = ax3.twinx()
        data_line, total_line = \
            ax3.plot(x_dates, stats['N_data'], 'g',
                     x_dates, stats['N_strike'] * stats['N_maturity'], 'k')
        perc_line, = ax4.plot(x_dates,
                              100 * stats['N_data'] / (stats['N_strike'] * stats['N_maturity']),
                             color='r')
        ax3.xaxis.grid(True)
        ax3.set_title('Sparsity of option price grid', fontsize=15)
        ax3.legend([data_line, total_line, perc_line],
                   ['Number of contracts', 'Price grid size', '% of grid filled'],
                   loc=0)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price Grid Count')
        ax4.set_ylabel('Percentage of Grid Filled (%)')
        fig.autofmt_xdate()

        plt.savefig('./data/stats_summary_' + _type + '.png', bbox_inches='tight')
        plt.show()

        stats.to_csv('./data/stats_summary_' + _type + '.csv')

def recovered_mean(path_to_file):
    recovered = pd.read_csv(path_to_file, index_col=0)
    recovered.columns = [float(_x) for _x in recovered.columns]
    recovered.index = [float(_x) for _x in recovered.index]

    # Integral and take average
    delta_K = recovered.columns[1] - recovered.columns[0]
    df_wts = recovered * delta_K * np.array(recovered.columns)
    return df_wts.sum(axis=1) / (recovered * delta_K).sum(axis=1)

def plot_matrix(path_to_file):
    mx = pd.read_csv(path_to_file, index_col=0)
    mx.T.plot()

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

def fitting_errors(option_type='call', bw=None):
    dates = priceMapLoader.getDates(option_type)
    err_metrics = pd.DataFrame(index=dates, columns=['N', 'RMSE', 'MAPE'])

    for _date in dates:
        df_real = priceMapLoader.loadDataOfDate(_date, option_type)
        try:
            df_fitted = interpSurface.obtainSurface(_date, df_real.columns, df_real.index,
                                                    option_type=option_type, bw=bw)[0]
        except FileNotFoundError:
            continue

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

        err_metrics.ix[_date, 'RMSE'] = np.sqrt(MSE / count)
        err_metrics.ix[_date, 'MAPE'] = MAPE / count
        err_metrics.ix[_date, 'N'] = count

    err_metrics.dropna(how='all', inplace=True)

    output_str = './data/error_metrics_'
    if bw is not None:
        output_str += 'bw_' + str(bw) + '_'
    output_str += option_type + '.csv'
    err_metrics.to_csv(output_str)

    return err_metrics

def fitting_violation(option_type='call', bw=400):
    dates = priceMapLoader.getDates(option_type)
    df_violation = pd.DataFrame(index=dates, columns=['N', 'count', 'percentage'])

    for _date in dates:
        try:
            name_str = './data/local_poly/no_fix_bw' + str(bw) + '_ms_call_' + str(_date) + '.csv'
            fitting = pd.read_csv(name_str)

        except OSError:
            continue

        df_violation.ix[_date, 'N'] = fitting.shape[0]
        df_violation.ix[_date, 'count'] = np.sum(np.any([fitting['beta0'] < 0,
                                                         fitting['beta1'] >= 0,
                                                         fitting['beta2'] <= 0],
                                                 axis=0))

    df_violation.dropna(how='all', inplace=True)
    df_violation.ix[:, 'percentage'] = df_violation['count'] / df_violation['N']

    df_violation.to_csv('./data/error_violations_bw_' + str(bw) + '_' + option_type +'.csv')
    return df_violation

def bw_error_plot(list_bws):
    fig = plt.figure()
    for _bw in list_bws:
        df_bw = pd.read_csv('./data/error_metrics_bw_' + str(_bw) + '_call.csv',
                            index_col=0)
        dates = pd.to_datetime([str(_date) for _date in df_bw.index])
        plt.plot(dates, df_bw['MAPE'] * 100)

    plt.title('Mean Absolute Percentage Errors for various bandwidth', fontsize=15)
    plt.ylabel("MAPE (%)")
    plt.xlabel("Date")
    plt.legend([str(_bw) for _bw in list_bws], loc=1)
    fig.autofmt_xdate()
    plt.savefig('./data/MAPE_bw.png', bbox_inches='tight')
    plt.show()

def bw_violation_plot(list_bws):
    fig = plt.figure()
    for _bw in list_bws:
        df_bw = pd.read_csv('./data/error_violations_bw_' + str(_bw) + '_call.csv',
                            index_col=0)
        dates = pd.to_datetime([str(_date) for _date in df_bw.index])
        plt.plot(dates, df_bw['percentage'] * 100)

    plt.title('Percentage non-arbitrage violations for various bandwidth', fontsize=15)
    plt.ylabel("Violations (%)")
    plt.xlabel("Date")
    plt.legend([str(_bw) for _bw in list_bws], loc=1)
    fig.autofmt_xdate()
    plt.savefig('./data/N_violations_bw.png', bbox_inches='tight')
    plt.show()
