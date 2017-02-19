# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:28:49 2017

@author: Yao Dong Yu

Description: clean up raw S&P 500 index and option data

Require: S&P 500 option data.csv
         SP500.csv
"""

import pandas as pd

raw_data = pd.read_csv("./data/S&P 500 option data.csv",
                       usecols=["secid", "date", "exdate", "cp_flag", "strike_price",
                                "best_bid", "best_offer", "volume", "impl_volatility",
                                "delta", "gamma", "vega", "theta", "exercise_style"])

# Remove any non European style options
raw_data.dropna(inplace=True, subset=["date", "exdate", "cp_flag", "strike_price",
                                      "best_bid", "best_offer", "impl_volatility",
                                      "exercise_style"])
raw_data = raw_data[raw_data["exercise_style"] == "E"]

date_start = pd.to_datetime(raw_data["date"], format="%Y%m%d")
date_end = pd.to_datetime(raw_data["exdate"], format="%Y%m%d")
raw_data["days_to_mature"] = (date_end - date_start).astype('timedelta64[D]').astype(int)

raw_call = raw_data[raw_data["cp_flag"] == "C"]
raw_put = raw_data[raw_data["cp_flag"] == "P"]

raw_call.to_csv("./data/raw_call.csv", index=False)
raw_put.to_csv("./data/raw_put.csv", index=False)

# Clean S&P 500 index data
SP500_index = pd.read_csv("./data/SP500.csv")
SP500_index.dropna(inplace=True)
SP500_index.reset_index(drop=True, inplace=True)
SP500_index["DATE"] = SP500_index["DATE"].map(lambda date_str: int(date_str.replace('-', '')))

SP500_index.to_csv("./data/SP500_clean.csv", index=False)