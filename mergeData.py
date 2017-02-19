# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:41:15 2017

@author: Yao Dong Yu

Description: merge daily price matrix within a certain time period, and separate
                 by index leve

Require: outputs of cleanData.py:
             SP500_clean.csv
         outputs of reshapeData.py:
             priceDfsMap_call.pickle
             priceDfsMap_put.pickle
"""

import os
import pickle
import pandas as pd

# Every 60 trading days
num_df_to_combine = 200
index_step_size = 5

def roundIndex(val):
    return int(val // index_step_size * index_step_size)

def saveMergedDFs(dict_df, date_init, option_type):
    with open('./data/merged/mergedDfsMap_all_' + option_type + '_' + str(date_init) + '.pickle', 'wb') as fp:
        pickle.dump(dict_df, fp)

def saveCleanMergedDFs(dict_df, date_init, option_type):
    with open('./data/merged/mergedDfsMap_all_clean_' + option_type + '_' + str(date_init) + '.pickle', 'wb') as fp:
        pickle.dump(dict_df, fp)

    for _index_level in dict_df:
        dict_df[_index_level]\
            .to_csv('./data/merged/mergedDfsMap_' + option_type + '_' + str(date_init) + '_' + str(_index_level) + '.csv')

SP500_index = pd.read_csv("./data/SP500_clean.csv")
SP500_index.set_index("DATE", inplace=True, drop=False)

for option_type in ["call", "put"]:
    with open('./data/priceDfsMap_' + option_type + '.pickle', 'rb') as fp:
        date_priceDf_map = pickle.load(fp)

    # Extract all dates in orders
    left_dates = set(SP500_index.index)
    right_dates = set(date_priceDf_map.keys())
    dates = list(left_dates & right_dates)
    dates.sort()

    dict_df_merged = dict()

    # Merging
    date_init = dates[0]
    dict_df_merged[date_init] = dict()
    for _idx, _date in enumerate(dates, 1):
        # after 300 iterations, store current result and re-initialize
        if not divmod(_idx, num_df_to_combine)[1]:
            # Save the merged map
            saveMergedDFs(dict_df_merged[date_init], date_init, option_type)

            date_init = _date
            dict_df_merged[date_init] = dict()
        else:
            index_key = roundIndex(SP500_index.ix[_date, "VALUE"])
            df_to_merge = date_priceDf_map[_date]
            if index_key in dict_df_merged[date_init]:
                dict_df_merged[date_init][index_key] = \
                    dict_df_merged[date_init][index_key].combine_first(df_to_merge)
            else:
                dict_df_merged[date_init][index_key] = df_to_merge

    # If not multiple, save the remaining
    if divmod(_idx, num_df_to_combine)[1]:
        saveMergedDFs(dict_df_merged[date_init], date_init, option_type)

# Reload, and match dimensions
for _filename in os.listdir('./data/merged/'):
    if _filename.startswith('mergedDfsMap_all_call_'):
        date_str = _filename[-15:-7]
        with open('./data/merged/' + _filename, 'rb') as fp:
            call_of_date = pickle.load(fp)
        with open('./data/merged/' + _filename.replace('call', 'put'), 'rb') as fp:
            put_of_date = pickle.load(fp)

        clean_call = dict()
        clean_put = dict()
        for _key in call_of_date:
            # must have same index levels
            if _key not in put_of_date:
                continue

            _call = call_of_date[_key]
            _put = put_of_date[_key]

            # Make sure y-axis(days till maturity) matches
            _index = _call.index.intersection(_put.index)
            # Make sure dimension of x-axis matches
            # ??does match strike value lose critical data??
            _columns = _call.columns.intersection(_put.columns)

            clean_call[_key] = _call.ix[_index, _columns]
            clean_put[_key] = _put.ix[_index, _columns]

        saveCleanMergedDFs(clean_call, date_str, 'call')
        saveCleanMergedDFs(clean_put, date_str, 'put')
