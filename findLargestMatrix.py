# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:11:11 2017

@author: kcfef

Description: utility function to create a date list ranked by number of option
                price data points of that date

Require: outputs of reshapeData.py:
             priceDfsMap_call.pickle
             priceDfsMap_put.pickle
"""

import pickle

size_ranks = dict()

for option_type in ['call', 'put']:
    with open('./data/priceDfsMap_' + option_type + '.pickle', 'rb') as fp:
        date_priceDf_map = pickle.load(fp)

    l_dates = list(date_priceDf_map.keys())
    l_dates.sort()
    l_sizes = [-1] * len(l_dates)

    for _idx, _date in enumerate(l_dates):
        _price_matrix = date_priceDf_map[_date]
        _N_valid = _price_matrix.notnull().sum().sum()
        l_sizes[_idx] = _N_valid

    ranked_by_size = list(zip(l_sizes, l_dates))
    ranked_by_size.sort(reverse=True)

    size_ranks[option_type] = ranked_by_size