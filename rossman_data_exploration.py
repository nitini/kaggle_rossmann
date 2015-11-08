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
import matplotlib
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
from scipy.stats import uniform

#%%
train_file = './train.csv'
test_file = './test.csv'
store_file = './store.csv'
output_file = './predictions.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
store = pd.read_csv(store_file)

#%% Cleaning up the data

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
    
    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)
    
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
    
features = []
build_features(features, train)
build_features([], test)

#%% Means grouped in different ways: 
store_means = train.groupby(['Store'])['Sales'].mean()
store_type_means = train.groupby(['StoreType'])['Sales'].mean()
store_by_dow_means = train.groupby(['Store','DayOfWeek'])['Sales'].mean()
store_by_day_means = train.groupby(['Store','day'])['Sales'].mean()
store_by_month_means = train.groupby(['Store','month'])['Sales'].mean()
store_by_year_means = train.groupby(['Store','year'])['Sales'].mean()


