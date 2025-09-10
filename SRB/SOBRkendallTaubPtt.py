#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
from collections import namedtuple


# Supporting Functions
# Data Preprocessing
def __preprocessing(x):
    x = np.asarray(x).astype(float)
    dim = x.ndim
    
    if dim == 1:
        c = 1
        
    elif dim == 2:
        (n, c) = x.shape
        
        if c == 1:
            dim = 1
            x = x.flatten()
         
    else:
        print('Please check your dataset.')
        
    return x, c

	
# Missing Values Analysis
def __missing_values_analysis(x, method = 'skip'):
    if method.lower() == 'skip':
        if x.ndim == 1:
            x = x[~np.isnan(x)]
            
        else:
            x = x[~np.isnan(x).any(axis=1)]
    
    n = len(x)
    
    return x, n

	
# ACF Calculation
def __acf(x, nlags):
    y = x - x.mean()
    n = len(x)
    d = n * np.ones(2 * n - 1)
    
    acov = (np.correlate(y, y, 'full') / d)[n - 1:]
    
    return acov[:nlags+1]/acov[0]


# vectorization approach to calculate mk score, S
def __mk_score(x, n):
    s = 0

    demo = np.ones(n) 
    for k in range(n-1):
        s = s + np.sum(demo[k+1:n][x[k+1:n] > x[k]]) - np.sum(demo[k+1:n][x[k+1:n] < x[k]])
        
    return s

	
# original Mann-Kendal's variance S calculation
def __variance_s(x, n):
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:            # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
        
    else:                 # there are some ties in data
        tp = np.zeros(unique_x.shape)
        demo = np.ones(n)
        
        for i in range(g):
            tp[i] = np.sum(demo[x == unique_x[i]])
            
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18
        
    return var_s


# standardized test statistic Z
def __z_score(s, var_s):
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    
    return z


# calculate the p_value
def __p_value(z, alpha):
    # two tail test
    p = 2*(1-norm.cdf(abs(z)))  
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    
    return p, h, trend


def __R(x):
    n = len(x)
    R = []
    
    for j in range(n):
        i = np.arange(n)
        s = np.sum(np.sign(x[j] - x[i]))
        R.extend([(n + 1 + s)/2])
    
    return np.asarray(R)


def __K(x,z):
    n = len(x)
    K = 0
    
    for i in range(n-1):
        j = np.arange(i,n)
        K = K + np.sum(np.sign((x[j] - x[i]) * (z[j] - z[i])))
    
    return K

	
# Original Sens Estimator
def __sens_estimator(x):
    idx = 0
    n = len(x)
    d = np.ones(int(n*(n-1)/2))

    for i in range(n-1):
        j = np.arange(i+1,n)
        d[idx : idx + len(j)] = (x[j] - x[i]) / (j - i)
        idx = idx + len(j)

    return d


def sens_slope(x):
    """
    This method proposed by Theil (1950) and Sen (1968) to estimate the magnitude of the monotonic trend. Intercept calculated using Conover, W.J. (1980) method.
    Input:
        x:   a one dimensional vector (list, numpy array or pandas series) data
    Output:
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line
    Examples
    --------
      >>> import numpy as np
	  >>> import pymannkendall as mk
      >>> x = np.random.rand(120)
      >>> slope,intercept = mk.sens_slope(x)
    """
    res = namedtuple('Sens_Slope_Test', ['slope','intercept'])
    x, c = __preprocessing(x)
#     x, n = __missing_values_analysis(x, method = 'skip')
    n = len(x)
    slope = np.nanmedian(__sens_estimator(x))
    intercept = np.nanmedian(x) - np.median(np.arange(n)[~np.isnan(x.flatten())]) * slope  # or median(x) - (n-1)/2 *slope
    
    return res(slope, intercept)


def seasonal_sens_slope(x_old, period=12):
    """
    This method proposed by Hipel (1994) to estimate the magnitude of the monotonic trend, when data has seasonal effects. Intercept calculated using Conover, W.J. (1980) method.
    Input:
        x:   a vector (list, numpy array or pandas series) data
		period: seasonal cycle. For monthly data it is 12, weekly data it is 52 (12 is the default)
    Output:
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line, where full period cycle consider as unit time step
    Examples
    --------
      >>> import numpy as np
	  >>> import pymannkendall as mk
      >>> x = np.random.rand(120)
      >>> slope,intercept = mk.seasonal_sens_slope(x, 12)
    """
    res = namedtuple('Seasonal_Sens_Slope_Test', ['slope','intercept'])
    x, c = __preprocessing(x_old)
    n = len(x)
    
    if x.ndim == 1:
        if np.mod(n,period) != 0:
            x = np.pad(x,(0,period - np.mod(n,period)), 'constant', constant_values=(np.nan,))

        x = x.reshape(int(len(x)/period),period)
    
#     x, n = __missing_values_analysis(x, method = 'skip')
    d = []
    
    for i in range(period):
        d.extend(__sens_estimator(x[:,i]))
        
    slope = np.nanmedian(np.asarray(d))
    intercept = np.nanmedian(x_old) - np.median(np.arange(x_old.size)[~np.isnan(x_old.flatten())]) / period * slope
    
    return res(slope, intercept)


def rank_estimator(x, p, var_s):
    x = np.array(x)  # Convert list to numpy array
    idx = 0
    n = len(x)
    d = np.ones(int(n*(n-1)/2))

    for i in range(n-1):
        j = np.arange(i+1,n)
        d[idx : idx + len(j)] = (x[j] - x[i]) / (j - i)
        idx = idx + len(j)

    if p < 1:
        C = norm.ppf(1 - (1-p )/ 2) * np.sqrt(var_s)
        k = idx
        rank_up = int(np.rint((k + C) / 2 + 1)) - 1  # Subtract 1 for 0-based indexing
        rank_lo = int(np.rint((k - C) / 2)) - 1  # Subtract 1 for 0-based indexing
        rank_d = np.sort(d)  
        lo = rank_d[rank_lo]
        up = rank_d[rank_up]
    else:
        rank_d = np.sort(d)
        lo = rank_d[0]
        up = rank_d[-1]

    return lo, up


def Dtau (x):
    # Assuming mean_discharge is a pandas Series
    row1 = x.index.values
    row2 = x.values

    # find ties
    ro1 = np.sort(row1)
    ro2 = np.sort(row2)
    bdiff = np.diff(np.unique(ro1, return_counts=True)[1])
    ediff = np.diff(np.unique(ro2, return_counts=True)[1])

    # correcting loss of first value using diff on with unique
    ta = bdiff[0] if bdiff[0] > 1 else 1
    tb = ediff[0] if ediff[0] > 1 else 1
    bdiff = np.concatenate(([ta], bdiff))
    ediff = np.concatenate(([tb], ediff))

    # Determine ties used for computing adjusted variance
    L1 = len(row1)
    tp = np.sum(bdiff * (bdiff - 1) * (2 * bdiff + 5))
    uq = np.sum(ediff * (ediff - 1) * (2 * ediff + 5))

    # adjustments when time index has multiple observations
    d1 = 9 * L1 * (L1-1) * (L1-2)
    tu1 = np.sum(bdiff * (bdiff - 1) * (bdiff - 2)) * np.sum(ediff * (ediff - 1) * (ediff - 2)) / d1
    d2 = 2 * L1 * (L1-1)
    tu2 = np.sum(bdiff * (bdiff - 1)) * np.sum(ediff * (ediff -1)) / d2

    # ties used for adjusting denominator in Tau
    t1a = (np.sum(bdiff * (bdiff - 1))) / 2
    t2a = (np.sum(ediff * (ediff - 1))) / 2

    # Calculate denominator with ties removed Tau-b
    D = np.sqrt(((.5*L1*(L1-1))-t1a)*((.5*L1*(L1-1))-t2a))
    # Calculation denominator no ties removed Tau
    Dall = L1 * (L1 - 1) / 2
    return Dall, D


	
def original_test(x_old, alpha = 0.05):
    """
    This function checks the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert 1987).
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (0.05 default)
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p-value of the significance test
        z: normalized test statistics
        Tau: Kendall Tau
        s: Mann-Kendal's score
        var_s: Variance S
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line
    Examples
    --------
	  >>> import numpy as np
      >>> import pymannkendall as mk
      >>> x = np.random.rand(1000)
      >>> trend,h,p,z,tau,s,var_s,slope,intercept = mk.original_test(x,0.05)
    """
    res = namedtuple('Mann_Kendall_Test', ['trend', 'h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept'])
    x, c = __preprocessing(x_old)
    x, n = __missing_values_analysis(x, method = 'skip')
    
    s = __mk_score(x, n)
    var_s = __variance_s(x, n)
    Tau = s/(.5*n*(n-1))
    
    z = __z_score(s, var_s)
    p, h, trend = __p_value(z, alpha)
    slope, intercept = sens_slope(x_old)

    return res(trend, h, p, z, Tau, s, var_s, slope, intercept)


def yue_wang_modification_test(x_old, alpha = 0.05, lag=None):
    
    res = namedtuple('Modified_Mann_Kendall_Test_Yue_Wang_Approach', ['trend', 'h', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept','lo','up','Taub'])
    x, c = __preprocessing(x_old)
    x, n = __missing_values_analysis(x, method = 'skip')
    xd = pd.Series(x)
    
    s = __mk_score(x, n)
    var_s = __variance_s(x, n)
    Dall,D = Dtau(xd)
    Tau = s/Dall
    Taub = s/D
    
    # Yue and Wang (2004) variance correction
    if lag is None:
        lag = n
    else:
        lag = lag + 1

    # detrending
    slope, intercept = sens_slope(x_old)
    x_detrend = x - np.arange(1,n+1) * slope
    
    # account for autocorrelation
    acf_1 = __acf(x_detrend, nlags=lag-1)
    idx = np.arange(1,lag)
    sni = np.sum((1 - idx/n) * acf_1[idx])
    
    n_ns = 1 + 2 * sni
    var_s = var_s * n_ns

    z = __z_score(s, var_s)
    p, h, trend = __p_value(z, alpha)
    lo, up = rank_estimator(x, p, var_s)

    return res(trend, h, p, z, Tau, s, var_s, slope, intercept,lo,up,Taub)

#Pettitt_test for change significance

def pettitt_dip(data):
    data = data[~np.isnan(data)]
    m = len(data)
    t1 = np.tile(data, (m, 1))
    t1 = t1.T
    v = np.sign(t1 - data)
    V = np.nansum(v, axis=1)
    U = np.cumsum(V)
    loc = np.argmax(np.abs(U))
    K = np.max(np.abs(U))
    pvalue = 2 * np.exp((-6 * K ** 2) / (m ** 3 + m ** 2))
    return np.array([loc, K, pvalue])

