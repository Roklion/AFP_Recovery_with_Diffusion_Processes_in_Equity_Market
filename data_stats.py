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
import os
from os.path import isfile, join

import loadDataOfDate as priceMapLoader
import interpolateSurface as interpSurface
import OU_simulation as OU_sim

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import plotConfig

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

def plotSurface(x, y, z, names=None, title="", save=False, save_to="./_temp.png"):
    x_mesh, y_mesh = np.meshgrid(np.array(x).astype(float), np.array(y).astype(float))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x_mesh, y_mesh, z.T, cmap=cm.coolwarm)

    ax.set_title(title, fontsize=15)
    if names is not None:
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])

    if save:
        plt.savefig(save_to, bbox_inches='tight')

    plt.show()

def plotSurfaces_P_Vs(date, delta=1000):
    SP500 = pd.read_csv("./data/SP500_clean.csv", index_col="DATE")
    if date in SP500.index:
        S0 = SP500.ix[date].values[0]
        Ks = np.arange(S0 - delta, S0 + delta, 25)
        ts = np.linspace(20, 600, 10)
        P, V, Vy, Vyy, Vt, _, _ = \
            interpSurface.obtainSurface(date, Ks, ts, option_type='call')

        fig = plt.figure()
        ax_P = fig.add_subplot(111, projection='3d')
        x = P.columns
        y = P.index
        x_mesh, y_mesh = np.meshgrid(np.array(x).astype(float), np.array(y).astype(float))
        ax_P.plot_surface(x_mesh, y_mesh, P.values, cmap=cm.coolwarm)
        ax_P.set_xlabel('Strike Price')
        ax_P.set_ylabel('Time to Maturity (days)')
        ax_P.set_zlabel('Price')
        ax_P.set_title('Interpolated Price surface', fontsize=15)
        plt.savefig("./data/interpolated_surface_" + str(date) + "_P.png", bbox_inches='tight')

        fig = plt.figure()
        ax_V = fig.add_subplot(111, projection='3d')
        x = V.columns
        y = V.index
        x_mesh, y_mesh = np.meshgrid(np.array(x).astype(float), np.array(y).astype(float))
        ax_V.plot_surface(x_mesh, y_mesh, V.values, cmap=cm.coolwarm)
        ax_V.set_xlabel('Strike Price')
        ax_V.set_ylabel('Time to Maturity (days)')
        ax_V.set_zlabel('State Price, V')
        ax_V.set_title('Interpolated State Price (V) surface', fontsize=15)
        plt.savefig("./data/interpolated_surface_" + str(date) + "_V.png", bbox_inches='tight')
        plt.show()

        fig = plt.figure()
        ax_Vy = fig.add_subplot(111, projection='3d')
        x = Vy.columns
        y = Vy.index
        x_mesh, y_mesh = np.meshgrid(np.array(x).astype(float), np.array(y).astype(float))
        ax_Vy.plot_surface(x_mesh, y_mesh, Vy.values, cmap=cm.coolwarm)
        ax_Vy.set_xlabel('Strike Price')
        ax_Vy.set_ylabel('Time to Maturity (days)')
        ax_Vy.set_zlabel('Vy')
        ax_Vy.set_title('Interpolated 1st-order of State Price (Vy) surface', fontsize=15)
        plt.savefig("./data/interpolated_surface_" + str(date) + "_Vy.png", bbox_inches='tight')
        plt.show()

        fig = plt.figure()
        ax_Vyy = fig.add_subplot(111, projection='3d')
        x = Vyy.columns
        y = Vyy.index
        x_mesh, y_mesh = np.meshgrid(np.array(x).astype(float), np.array(y).astype(float))
        ax_Vyy.plot_surface(x_mesh, y_mesh, Vy.values, cmap=cm.coolwarm)
        ax_Vyy.set_xlabel('Strike Price')
        ax_Vyy.set_ylabel('Time to Maturity (days)')
        ax_Vyy.set_zlabel('Vyy')
        ax_Vyy.set_title('Interpolated 2nd-order of State Price (Vyy) surface', fontsize=15)
        plt.savefig("./data/interpolated_surface_" + str(date) + "_Vyy.png", bbox_inches='tight')
        plt.show()


    else:
        print("Date not found in data")


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

def average_distribution(eta, S0):
    sim_path = './data/sim_pipline_output/'
    files = os.listdir(sim_path)
    files = [_f for _f in files if isfile(join(sim_path, _f)) \
                                   and ('sim_test_S' + str(S0) +'_eta'+str(eta)) in _f \
                                   and '_adj_recovered.csv' in _f]

    distributions = list()
    ts = None
    Ks = None
    for _f in files:
        df = pd.read_csv(sim_path + _f, index_col=0)
        distributions.append(df)

        if ts is None:
            ts = df.index
        else:
            ts = ts.intersection(df.index)

        if Ks is None:
            Ks = df.columns
        else:
            Ks = Ks.intersection(df.columns)

    for _i, _ in enumerate(distributions):
        distributions[_i] = distributions[_i].ix[ts, Ks].values

    distributions = np.array(distributions)
    mean_dist = distributions.mean(axis=0)

    df_mean = pd.DataFrame(mean_dist, index=ts, columns=Ks)

    return df_mean

def plot_2d_siyuan(V, S0, title='', save_to=None):
    cm_subsection = np.linspace(0.0, 1, V.shape[0])
    colors = [cm.coolwarm(x) for x in cm_subsection]

    plt.figure(figsize=[8,5], facecolor='white')
    for i in range(V.shape[0]):
        if i==0 or i== (V.shape[0]-1):
            plt.plot(V.columns/S0, V.iloc[i,:],c=colors[i],lw=1.5, label='T='+str(V.index[i]))
        else:
            plt.plot(V.columns/S0, V.iloc[i,:],c=colors[i],lw=1.5, label='_nolegend_')
    plt.xlim([0.0,2.0])
    plt.legend(fontsize=15)
    plt.title(title, fontsize=15)
    plt.xlabel('Moneyness')
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()

def simulation_error(S0, etas, sims):
    df_RMSE = pd.DataFrame(index=sims, columns=etas)

    for _eta in etas:
        for _sim in sims:
            try:
                df_recovered = pd.read_csv("./data/sim_pipline_output/sim_test_S" + \
                                           str(S0) + "_eta" + str(_eta) + "_" + \
                                           str(_sim) + "_adj_recovered.csv",
                                           index_col=0)
            except OSError:
                continue

            ts = np.array([float(_index) for _index in df_recovered.index])
            Ks = np.array([float(_col) for _col in df_recovered.columns])
            df_recovered.index = ts
            df_recovered.columns = Ks

            _error = PDF_error(df_recovered, S0).ix[1.8]
            #_RMSE = np.sqrt(np.sum(np.square(df_recovered.values - df_true.values)) / df_true.size)
            df_RMSE.ix[_sim, _eta] = _error

    df_RMSE.to_csv("./data/sim_pipline_output/error.csv")
    return df_RMSE

def simulation_error_Vyy(S0, etas, sims):
    df_RMSE = pd.DataFrame(index=sims, columns=etas)

    for _eta in etas:
        for _sim in sims:
            try:
                df_recovered = pd.read_csv("./data/sim_pipline_output/sim_test_S" + \
                                           str(S0) + "_eta" + str(_eta) + "_" + \
                                           str(_sim) + "_Vyy.csv",
                                           index_col=0)
            except OSError:
                continue

            ts = np.array([float(_index) for _index in df_recovered.index])
            Ks = np.array([float(_col) for _col in df_recovered.columns])
            df_recovered.index = ts
            df_recovered.columns = Ks

            df_true = OU_sim.Get_Four(Ks, ts, 0.01, 0.001, S0, 1, 2100, .3 * S0, 0.1,
                                      lambda _x: (_x/2000 + 1)**-3)[3]
            _RMSE = ((df_recovered-df_true).abs()).sum(axis=1)*(Ks[1] - Ks[0])
            df_RMSE.ix[_sim, _eta] = _RMSE.ix[1.8]

    df_RMSE.to_csv("./data/sim_pipline_output/error_Vyy.csv")
    return df_RMSE

def PDF_error(f, S0):
    K=np.array(f.columns.tolist())
    T=np.array(f.index.tolist())
    real=OU_sim.OU_density(S0, 1, 2100, .3 * S0, K ,T)
    real.columns=K
    error=((f-real).abs()).sum(axis=1)*(K[1]-K[0])
    return error

def box_eta_error(df_error, title='', save_to=None):
    plt.figure(figsize=[8,5], facecolor='white')
    plt.boxplot([list(df_error.ix[df_error[_col].notnull(), _col]) for _col in df_error.columns],
                 sym='o')
    plt.axhline(0.03167, lw=1, color='r')
    plt.xticks(np.arange(len(df_error.columns))+1, df_error.columns)

    plt.title(title, fontsize=15)
    plt.xlabel('eta')
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()

