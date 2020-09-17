
"""

    Time Series Modeling/Assessment

    Transforms historical nickel price data and uses it to build various supervised learning regression
    
    models to make price forecasts for one year ahead of time. Predictor variables are created by
    
    taking lagged copies of itself and a target variable, y, which represents one year ahead of time,
    
    can be built by shifting all other X variables back by one year (261 business days). 


    To compare with baseline models, models are also fit on data with one day ahead target variables.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import sys
import argparse
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.ensemble import AdaBoostRegressor

from pylab import rcParams
rcParams['figure.figsize'] = 9, 6
import ast
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def lme_clean(filepath):
    """Reads source file, cleans, and reformats data as a time series in business days (B) as time units"""
    
    LME_futures = pd.read_excel(filepath)
    LME_futures = LME_futures.iloc[3:, 1:]
    LME_futures.columns = ['Date', 'Cash Price ($/MT)', 'Inventory (MT)']
    LME_futures.index = LME_futures['Date']
    LME_futures = LME_futures[LME_futures.index.year>=2005] # For now only use years after 2005

    LME_futures = LME_futures.iloc[:, 1:]

    LME = LME_futures.iloc[:, 0]
    LME = LME.astype(float)
    LME = LME.resample('B').mean()
    LME = LME.squeeze()
    return LME
