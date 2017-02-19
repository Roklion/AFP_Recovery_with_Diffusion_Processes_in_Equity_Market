# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:28:49 2017

@author: Yao Dong Yu

Description: re-organize option data into daily price matrix

Require: outputs of cleanData.py:
             raw_call.csv
             raw_put.csv
"""

import pandas as pd
import pickle

def constructPriceMatrix(df):
    df['strike_price'] = df['strike_price'] / 1000

    # Obtain x-axis Strike and y-axis Time-to-Maturity
    strikes = df["strike_price"].unique()
    strikes.sort()
    time_to_maturity = df["days_to_mature"].unique()
    time_to_maturity.sort()

    # Create Dataframe with x/y axis index
    df_mkt_price = pd.DataFrame(index=time_to_maturity, columns=strikes)

    # Fill in data
    df.apply(lambda _row: fillPriceDf(_row, df_mkt_price), axis=1)

    return df_mkt_price

def fillPriceDf(row_data, priceDf):
    col = row_data["strike_price"]
    idx = row_data["days_to_mature"]

    priceDf.ix[idx, col] = (row_data["best_bid"] + row_data["best_offer"]) / 2

# Clean price matrix based on method of Bakshi, Cao and Chen (2000)
def cleanPriceMatrix(priceDf):
    # Only use contracts with maturity time greater than 6 days
    out_df = priceDf.ix[priceDf.index > 6, :]
    # remove all entries with price less than $3/8
    out_df = out_df[out_df >= 3/8]

    #clean up
    out_df.dropna(axis=(0, 1), how='all', inplace=True)

    return out_df

for option_type in ["call", "put"]:
    # Read clean data
    raw_data = pd.read_csv("./data/raw_" + option_type + ".csv")

    map_priceDfs = dict()
    # For each date, create price matrix of time to maturity vs. strike
    for _date, _df_of_date in raw_data.groupby(["date"]):
        print(_date)
        _p_mx = constructPriceMatrix(_df_of_date)
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
