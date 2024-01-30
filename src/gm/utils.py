# -*- coding: utf-8 -*-
"""
utils.py

Description.

Author: drotto
Created: 1/29/24 @ 10:25
Project: gm
"""

import pandas as pd
import numpy as np
import scipy as sci

#%%

# preallocate empty array and assign slice by chrisaycock
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


# relevant as of Numpy 1.24
# https://github.com/iperov/DeepFaceLab/pull/5618/commits/52dcf152e2b5aedf96b5c8a343cad24585492df6
def fast_clip(array, min_value, max_value):
    return np.minimum(max_value, np.maximum(array, min_value, out=array), out=array)


def std_cinterval(d, a):
    '''

    :param d: data
    :type d:
    :param a: confidence level
    :type a:
    :return:
    :rtype:
    '''
    dof = len(d) - 1
    lower = np.sqrt((dof * d.std() ** 2) / sci.stats.chi2.ppf((a) / 2, df=dof))
    upper = np.sqrt((dof * d.std() ** 2) / sci.stats.chi2.ppf((1 - a) / 2, df=dof))
    return lower, upper


def autocorr(x, t):
    return np.corrcoef(np.array([x[:-t], x[t:]]))


def autocorr2(x, t, mean, var):
    x -= mean
    return (x[: x.size - t] * x[t:]).mean() / var


def acf(x, t):
    return np.array([autocorr2(x.copy(), i, x.mean(), x.var()) for i in range(t)])
