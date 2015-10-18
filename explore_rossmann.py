#%% Import necessary models
import numpy as np
import csv
import time
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib

#%% Loading in the data
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

#%% Evaluation Metric Functions
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe
    
def rmspe_cust(yhat, y):
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w* (y - yhat)**2))
    return rmspe


#%% For XGBoost, make features all numeric
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

#%% Creating the feature set
features = []
build_features(features, train)
build_features([], test)

#%% Training the XGBoost Model (Using XGBoost library)
params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }
num_trees = 300

val_size = 100000
X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train['Sales'] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test['Sales'] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist,
                early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)
                

# Validation
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)

# Make Predictions
test_probs = gbm.predict(xgb.DMatrix(test[features]), 
                         ntree_limit=gbm.best_iteration)
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({'Id': test['Id'], 'Sales': np.exp(test_probs) - 1})
submission.to_csv('ni_xgboost_submission_10152015.csv', index=False)

#%% Training the XGBoost Model (Using sklearn)
X_train, X_val = cross_validation.train_test_split(train, 
                                                   test_size=0.01)
y_val = X_val['Sales']                                              

X_test = test[features]




xgb_params = {'loss':'ls',
              'n_estimators': 300,
              'max_depth': 5,
              'lr': 0.1,
              'max_features': 'auto',
              'subsample':1.0,
              'verbose':1}

xgb_train = GradientBoostingRegressor(loss=xgb_params['loss'],
                                      n_estimators=xgb_params['n_estimators'], 
                                      max_depth=xgb_params['max_depth'], 
                                      learning_rate=xgb_params['lr'],
                                      max_features=xgb_params['max_features'],
                                      subsample=xgb_params['subsample'],
                                      verbose=xgb_params['verbose'])
                                              
#%% Training the model
xgb_train.fit(X_train[features], np.log(X_train['Sales'] + 1))

#%% Validation Error
pred_val_probs = xgb_train.predict(X_val[features])
indices = pred_val_probs < 0 
pred_val_probs[indices] = 0
val_rmspe = rmspe(np.exp(pred_val_probs) - 1, y_val.values)
print('Validation Set Error: ' + val_rmspe)

#%% Grid search for sklearn Gradient Boosting






#%% Make Predictions
pred_test_probs = xgb_train.predict(X_test)

indices = pred_test_probs < 0
pred_test_probs[indices] = 0
submission = pd.DataFrame({'Id': test['Id'], 
                           'Sales': np.exp(pred_test_probs) - 1})
submission.to_csv('ni_sklearn_xgb_10172015_v2.csv', index=False)




    
