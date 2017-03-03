# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:02:04 2017

@author: Yao Dong Yu

Description: script to generate recovery from interpolated surface out of real data

Require: outputs of interpolateSurface.py: formatPoly_main()
            surface_map/surface_poly_call_*.pickle
            surface_map/surface_poly_put_*.pickle
"""


import pandas as pd
import numpy as np
import interpolateSurface
import continuous_recovery

delta = 500
option_type = 'call'
date_range = [20100101, 20141231]
ts = np.linspace(150, 500, 10)

SP500 = pd.read_csv("./data/SP500_clean.csv", index_col="DATE")

for _date_str in SP500.index:
    _date = int(_date_str)
    if(_date >= date_range[0] and _date <= date_range[1]):
        print("Date=" + str(_date))
        try:
            index_val = SP500.ix[_date].values[0]
            _Ks = np.linspace(index_val - delta, index_val + delta, 50)
            Price, V, Vy, Vyy, Vt, t_out, K_out = interpolateSurface.obtainSurface(_date, _Ks, ts, option_type)
            suffix = "_" + str(_date) + ".csv"
            Price.to_csv("./data/surface_map/Price" + suffix)
            V.to_csv("./data/surface_map/V" + suffix)
            Vy.to_csv("./data/surface_map/Vy" + suffix)
            Vyy.to_csv("./data/surface_map/Vyy" + suffix)
            Vt.to_csv("./data/surface_map/Vt" + suffix)

            recovered, m_hat, rho, S0, V_new, FoundPositive = \
                continuous_recovery.RECOVERY(V, Vy, Vyy, Vt, SP500.ix[_date].values[0])

            recovered.to_csv("./data/surface_map/z_recovered_" + str(_date) + ".csv")
        except RuntimeError as err:
            print("Recovery Failed: ", err)


