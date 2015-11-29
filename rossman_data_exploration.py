# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:14:29 2015

@author: nitini
"""
#%%
import numpy as np
import csv
import time
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
from scipy.stats import uniform
from ggplot import *
#%% Loading in the data and cleaning it up
train_file = './train.csv'
test_file = './test.csv'
store_file = './store.csv'
output_file = './predictions.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
store = pd.read_csv(store_file)



# Assume store open if not known
test.fillna(1, inplace=True)

# Consider only open stores, closed don't count
train = train[train['Open'] != 0]

# Add store features to training data
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

def build_features(features, data):
    
    # Remove NaNs
    data.fillna(0,inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    
    # No need to process some features
    features.extend(['Store', 'CompetitionDistance', 
                     'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2',
                     'Promo2SinceWeek', 'Promo2SinceYear'])
    
    # Preprocess more features
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    
    features.append('dayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    features.append('weekOfYear')
    
    data['Date'] = pd.to_datetime(data['Date'])
    data['year'] = data.Date.dt.year
    data['day'] = data.Date.dt.day
    data['month'] = data.Date.dt.month
    data['dayOfWeek'] = data.Date.dt.dayofweek
    data['weekOfYear'] = data.Date.dt.weekofyear
    
    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)
    
    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)
    
     #Code to make a promo binary variable 
    #(Is there a promo running on day X). Needs to be fixed still
    
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + \
        (data.weekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Okt', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) &
                (data.PromoInterval == interval), 'IsPromoMonth'] = 1

features = []
build_features(features, train)
build_features([], test)

#%% Calculate mean sales for each store and join to training data
store_means = pd.DataFrame(pd.Series(train.groupby(['Store'])['Sales'].mean(),
                        name = 'store_sales_mean'))
train = pd.merge(train, store_means, left_on='Store', right_index='True')
features.append('store_sales_mean')

#%% Calculate mean sales for each store for each day of week, join to train
store_by_dow_means = train.groupby(['Store','DayOfWeek'])['Sales'].mean()
store_by_dow_means.index.levels[0].name = 'store_index'
store_by_dow_means.index.levels[1].name = 'dow'
store_by_dow_means = pd.DataFrame(pd.Series(store_by_dow_means, 
                                name='store_dow_sales_means'))
train = train.join(store_by_dow_means, on=['Store','DayOfWeek'])

#%%
store_by_day_means = train.groupby(['Store','day'])['Sales'].mean()
store_by_month_means = train.groupby(['Store','month'])['Sales'].mean()
store_by_year_means = train.groupby(['Store','year'])['Sales'].mean()
store_type_means = train.groupby(['StoreType'])['Sales'].mean()
#%% Joining means to the training data.

#%% Implementing ggplot to look at the data

train_sales = pd.DataFrame(train[train['Store'] == 1]['Sales'])

ggplot(aes(x='Sales'), data=train_sales) + \
    geom_histogram()




