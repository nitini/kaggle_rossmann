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
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt

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
features.append('store_dow_sales_means')

#%% Implementing ggplot to look at the data

train_sales = pd.DataFrame(train[train['Store'] == 1]['Sales'])

ggplot(aes(x='Sales'), data=train_sales) + \
    geom_histogram()
    
store_type_means = pd.DataFrame(train.groupby(['StoreType'])['Sales'].mean())

store_type_means['StoreType'] = pd.Series(['A','B','C','D'], 
                                            index=store_type_means.index)
store_type_means.head()

ggplot(aes(x='StoreType', y='Sales'), data=store_type_means) + \
     geom_bar(stat="identity")
     
ggplot(aes(x='Sales'), data=train) + \
    geom_histogram() + facet_wrap('DayOfWeek')
    
ggplot(aes(x='Sales'), data=train) + \
    geom_histogram() + facet_wrap('IsPromoMonth')
    
promo_means = train.groupby(['year', 'IsPromoMonth'])['Sales'].mean()
promo_means = pd.DataFrame(promo_means)

day_means = train.groupby(['day'])['Sales'].mean()
month_means = train.groupby(['month'])['Sales'].mean()
plt.plot(day_means)
plt.plot(month_means)

store_num = 979
specific_store_sales = train[(train.Store==store_num)][['Date',
                                                        'month',
                                                        'Sales', 
                                                        'year',
                                                        'DayOfWeek']]
ggplot(aes(x='Date', y='Sales', color='DayOfWeek'), 
       data=specific_store_sales) +\
    geom_line()

#%% RMSPE Function

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

#%% Testing out Vanilla K-Fold Cross Validation on Training Data

xgb_params = {'loss':'ls',
              'n_estimators': 10,
              'max_depth': 8,
              'lr': 0.1,
              'max_features': 'auto',
              'subsample':1.0,
              'verbose':1}
              
xgb_train = GradientBoostingRegressor(loss=xgb_params['loss'],
                                      n_estimators=
                                      xgb_params['n_estimators'], 
                                      max_depth=
                                      xgb_params['max_depth'], 
                                      learning_rate=xgb_params['lr'],
                                      max_features=
                                      xgb_params['max_features'],
                                      subsample=xgb_params['subsample'],
                                      verbose=xgb_params['verbose'])

cross_val_scores = []
kf = KFold(train.shape[0], n_folds=3)
fold_counter = 0

for train_indices, test_indices in kf:
    fold_counter += 1
    print('Processing fold: ' + str(fold_counter))
    fold_X_train = train[features].iloc[train_indices]
    fold_y_train = train['Sales'].iloc[train_indices]
    fold_X_test = train[features].iloc[test_indices]
    fold_y_test = train['Sales'].iloc[test_indices]
    xgb_train.fit(fold_X_train, np.log(fold_y_train + 1))
    pred_val_probs = xgb_train.predict(fold_X_test)
    indices = pred_val_probs < 0 
    pred_val_probs[indices] = 0
    val_rmspe = rmspe(np.exp(pred_val_probs) - 1, fold_y_test.values)
    cross_val_scores.append(val_rmspe)

#%% Function that takes a single time series and creates appropriate CV folds
def performTimeSeriesCV(X_train, y_train, number_folds):
    print 'Size train set: ', X_train.shape

    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    print 'Size of each fold: ', k

    for i in range(2, number_folds + 1):
        print ''
        
        split = float(i-1)/i
        
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        
        index = int(np.floor(X.shape[0] * split))
      
        X_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        print 'Size of train: ', X_trainFolds.shape

        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]
        
        print 'Size of test: ', X_testFold.shape

#%% Building out Time Series CV for all training data using above function
