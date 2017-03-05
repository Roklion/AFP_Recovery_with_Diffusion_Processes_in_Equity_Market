# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:27:13 2017

@author: Yao Dong Yu

Description: script to generate results for interpolating errors
"""

import constrainedTransform as transform
import interpolateSurface as interpAPI
import data_stats as interpStats

def bw_analysis_main(list_bws):
    for _bw in list_bws:
        interpAPI.localPolySmoother_main(option_type='call', bandwidth=_bw)

        interpAPI.formatPoly_main(option_type='call', bw=_bw)

        _ = interpStats.fitting_errors(option_type='call', bw=_bw)


def transform_analysis_main(bw=400):
    transform.transform_main(no_transform=True, date_range=[20140101, 20161231])

    interpAPI.localPolySmoother_main(option_type='call', bandwidth=bw, no_transform=True)
    interpAPI.formatPoly_main(option_type='call', bw=bw, no_transform=True, )

def violation_analysis_main(list_bws):
    for _bw in list_bws:
        interpAPI.localPolySmoother_main(option_type='call', bandwidth=_bw,
                                         no_transform=False, fix_violation=False)

        _ = interpStats.fitting_violation(option_type='call', bw=_bw)


