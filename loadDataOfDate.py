# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:51:42 2017

@author: kcfef

Description: utility function to load price matrix of a specific day
                and option type

Require: outputs of reshapeData.py:
             priceDfsMap_call.pickle
             priceDfsMap_put.pickle
"""

import pickle

# Load date to price dataframe map if not already loaded
try:
    date_priceDf_map_call
except NameError:
    with open('./data/priceDfsMap_call.pickle', 'rb') as fp:
        date_priceDf_map_call = pickle.load(fp)

try:
    date_priceDf_map_put
except NameError:
    with open('./data/priceDfsMap_put.pickle', 'rb') as fp:
        date_priceDf_map_put = pickle.load(fp)

#==============================================================================
# loadDataOfDate
#
# given date index (integer, format of yyyymmdd), return the price dataframe
# with index of days till maturity, column names of strike price * 1000
#==============================================================================
def loadDataOfDate(date, option_type="call"):
    global date_priceDf_map_call
    global date_priceDf_map_put

    try:
        if option_type == "call":
            return date_priceDf_map_call[date]
        else: #put
            return date_priceDf_map_put[date]
    except KeyError:
        print("No pricing entry of date: " + str(date) + " for " + option_type + " option")
        return None