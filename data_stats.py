# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:00:33 2017

@author: Yao Dong Yu

Description: sumary statistics and plot of cleaned data

Require: outputs of reshapeData.py:
             priceDfsMap_call.pickle
             priceDfsMap_put.pickle
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt

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

data_stats = dict()
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
