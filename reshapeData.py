# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:28:49 2017

@author: Yao Dong Yu

Description: re-organize option data into daily price matrix

Require: outputs of cleanData.py:
             raw_call.csv
             raw_put.csv
"""

import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize

import loadDataOfDate as priceMapLoader

def constructPriceMatrix(df, fill_func):
    # Obtain x-axis Strike and y-axis Time-to-Maturity
    strikes = df["strike_price"].unique()
    strikes.sort()
    time_to_maturity = df["days_to_mature"].unique()
    time_to_maturity.sort()

    # Create Dataframe with x/y axis index
    df_mkt_price = pd.DataFrame(index=time_to_maturity, columns=strikes)

    # Fill in data
    df.apply(lambda _row: fill_func(_row, df_mkt_price), axis=1)

    return df_mkt_price

def fillPriceDf(row_data, priceDf):
    col = row_data["strike_price"]
    idx = row_data["days_to_mature"]

    priceDf.ix[idx, col] = (row_data["best_bid"] + row_data["best_offer"]) / 2

def fillPriceDf_sim(row_data, priceDf):
    col = row_data["strike_price"]
    idx = row_data["days_to_mature"]

    priceDf.ix[idx, col] = (row_data["option_price"])

# Clean price matrix based on method of Bakshi, Cao and Chen (2000)
def cleanPriceMatrix(priceDf):
    # Only use contracts with maturity time greater than 6 days
    out_df = priceDf.ix[priceDf.index > 6, :]
    # remove all entries with price less than $3/8
    out_df = out_df[out_df >= 3/8]

    #clean up
    out_df.dropna(axis=(0, 1), how='all', inplace=True)

    return out_df

def reshapeData_main():
    for option_type in ["call", "put"]:
        # Read clean data
        raw_data = pd.read_csv("./data/raw_" + option_type + ".csv")

        map_priceDfs = dict()
        # For each date, create price matrix of time to maturity vs. strike
        for _date, _df_of_date in raw_data.groupby(["date"]):
            print(_date)
            # Real data has strike price * 1000
            _df_of_date['strike_price'] = _df_of_date['strike_price'] / 1000
            _p_mx = constructPriceMatrix(_df_of_date, fillPriceDf)
            _p_mx = cleanPriceMatrix(_p_mx)
            if _p_mx.size > 0:
                map_priceDfs[_date] = _p_mx.copy()

        with open('./data/priceDfsMap_' + option_type + '.pickle', 'wb') as fp:
            pickle.dump(map_priceDfs, fp)

#==============================================================================
# Example of loading data:
#
# import pickle
# with open('./data/priceDfsMap_call.pickle', 'rb') as fp:
#     date_priceDf_map_call = pickle.load(fp)
#
#==============================================================================

def mergeCallPut_main():
    t_bound = 6
    output_map = dict()

    SP500 = pd.read_csv("./data/SP500_clean.csv", index_col="DATE")
    SP500_dates = np.array(SP500.index).astype(int)

    call_dates = np.intersect1d(SP500_dates, np.array(priceMapLoader.getDates('call')))
    put_dates = np.intersect1d(SP500_dates, np.array(priceMapLoader.getDates('put')))

    # Matrix only exist for call or put
    only_call = np.setdiff1d(call_dates, put_dates)
    for _date in only_call:
        output_map[_date] = priceMapLoader.loadDataOfDate(_date, 'call')
    only_put = np.setdiff1d(put_dates, call_dates)
    for _date in only_put:
        output_map[_date] = priceMapLoader.loadDataOfDate(_date, 'put')

    # Intersection dates
    dates_both = np.intersect1d(call_dates, put_dates)
    for _date in dates_both:
        print(_date)
        _df_call = priceMapLoader.loadDataOfDate(_date, 'call')
        _df_put = priceMapLoader.loadDataOfDate(_date, 'put')
        _S0 = SP500.ix[_date, :].values[0]

        # Use intersected area to derive put-call parity d and r
        _inters_t = np.intersect1d(_df_call.index, _df_put.index)
        _all_Ks = np.union1d(_df_call.columns, _df_put.columns)

        _df_overlap_c = _df_call.ix[_inters_t, :]
        _df_overlap_p = _df_put.ix[_inters_t, :]

        _df_new_Cs = pd.DataFrame(index=_inters_t, columns=_all_Ks, dtype=float)
        # one put-call partiy each time to maturity
        for _t in _inters_t:
            if _t < t_bound:
                continue

            _C_t = _df_overlap_c.ix[_t, :].dropna(how='any')
            _P_t = _df_overlap_p.ix[_t, :].dropna(how='any')

            _inters_K = _C_t.index.intersection(_P_t.index)
            _overlap_c_t = _C_t[_inters_K]
            _overlap_p_t = _P_t[_inters_K]
            _Cs_minus_Ps = _overlap_c_t - _overlap_p_t

            _x0 = (_S0, 0.05)
            _bnds = ((0, None), (0, None))
            _res = minimize(putCallParityObjective, _x0, method='SLSQP', bounds=_bnds,
                            args=(_Cs_minus_Ps, np.array(_inters_K)))
            _xs = _res.x
            #_log_xs = np.log(_xs) / (-_t / 365)
            _F = _xs[0]
            _D = _xs[1]
            #print(_t, _F, _D)
            #print(_res.fun)

            _df_Cs = pd.DataFrame(index=_C_t.index.union(_P_t.index),
                                  columns=['C', 'calc C', 'mean'], dtype=float)

            _df_Cs.ix[_C_t.index, 'C'] = _C_t.values
            _df_Cs.ix[_P_t.index, 'calc C'] = \
                getCall_PCParity(_P_t.values, _F, np.array(_P_t.index), _D)

            # Ignore negative number, and calculate mean when possible
            _df_Cs[_df_Cs <= 0] = np.nan
            _df_Cs = _df_Cs.dropna(how='all', axis=0)

            _df_Cs.ix[:, 'mean'] = np.nanmean(_df_Cs.ix[:, ['C', 'calc C']],
                                              axis=1)

            #print(_df_Cs)

            _df_new_Cs.ix[_t, _df_Cs.index] = _df_Cs.ix[:, 'mean']

        # Clean up mean Cs from put-call partiy
        _df_new_Cs.dropna(how='all', axis=0, inplace=True)
        _df_new_Cs.dropna(how='all', axis=1, inplace=True)

        output_map[_date] = _df_new_Cs

    with open('./data/priceDfsMap_merged_call.pickle', 'wb') as fp:
        pickle.dump(output_map, fp)

def putCallParityObjective(xs, Cs_minus_Ps, Ks):
    return np.sum(np.square(Cs_minus_Ps - xs[0] + Ks * xs[1]))

def getCall_PCParity(Ps, F, Ks, D):
    return Ps + F - Ks * D

def getPut_PCParity(Cs, F, Ks, D):
    return Cs - F + Ks * D