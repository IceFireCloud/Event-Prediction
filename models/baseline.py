'''
baseline.py
Contians the baseline for EP project
A model is meaningful only if it's performance is better than baseline
'''
import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
from data import process_data
from random import random

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

csvfilename = '/scratch2/xzhou/ep/LA_crime/data/Block_Columns.csv'
categorical = False

# Simply computes the average in training data
def naive_average(block_num):
    #loss is computed per prediction, the sum is not calculated
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    print('Average method:', average)
    if categorical:
        test_y = np.argmax(test_y, axis=1)
        print('categorical rmse:', np.sqrt(np.sum( (np.array(int(average)) - test_y)**2 )))
    else:
        print('numerical rmse:', np.sqrt(np.sum( (average - test_y)**2 )))
    return

# Simple MLP
def naive_MLP(block_num):
    #loss is computed per prediction, the sum is not calculated
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    print('2-layer MLP method:')
    model = Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    if categorical:
        model.add(layers.Dense(train_y.shape[1], activation=None))
    else:
        model.add(layers.Dense(1, activation=None))
    model.compile(optimizer='rmsprop', loss='mae')
    model.fit(train_x, train_y, epochs=10, verbose=0)
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    
    if categorical:
        test_y = np.argmax(test_y, axis=1)
        prediction = np.argmax(prediction, axis=1)
        print('categorical rmse:', np.sqrt(np.sum((prediction - test_y)**2 )))
    else:
        print('numerical rmse:', np.sqrt(np.sum( (prediction - test_y)**2 )))
    return

# Simple regression methods
def naive_regression(block_num):
    #loss is computed per prediction, the sum is not calculated
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    print('regression methods:')
    train_x = np.reshape(train_x, (train_x.shape[0], -1))
    test_x = np.reshape(test_x, (test_x.shape[0], -1))
    if categorical:
        train_y = np.argmax(train_y, axis=1)
        test_y = np.argmax(test_y, axis=1)

        reg1 = LinearRegression().fit(train_x, train_y)
        print('Linear Regression score:', reg1.score(test_x, test_y))
        pred_y = reg1.predict(test_x).astype(int)
        print('Linear Regression categorical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))
        
        #hard to converge, commenting out for now
        # reg2 = LogisticRegression().fit(train_x, train_y)
        # print('Logistic Regression score:', reg2.score(test_x, test_y))
        # pred_y = reg2.predict(test_x).astype(int)
        # print('Logistic Regression categorical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))
    else:
        reg1 = LinearRegression().fit(train_x, train_y)
        print('Linear Regression score:', reg1.score(test_x, test_y))
        pred_y = reg1.predict(test_x).astype(int)
        print('Linear Regression numerical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))
        
        # reg2 = LogisticRegression().fit(train_x, train_y)
        # print('Logistic Regression score:', reg2.score(test_x, test_y))
        # pred_y = reg2.predict(test_x).astype(int)
        # print('Logistic Regression numerical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2)))
    return

# Gradient Boosting method
def gradient_boost(block_num):
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    print('Gradient Boosting method:')
    train_x = np.reshape(train_x, (train_x.shape[0], -1))
    test_x = np.reshape(test_x, (test_x.shape[0], -1))
    clf = GradientBoostingClassifier(random_state=0)
    if categorical:
        train_y = np.argmax(train_y, axis=1)
        test_y = np.argmax(test_y, axis=1)

        clf.fit(train_x, train_y)
        print('Gradient Boosting score:', clf.score(test_x, test_y))
        pred_y = clf.predict(test_x).astype(int)
        print('Gradient Boosting categorical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))
    else:
        clf.fit(train_x, train_y)
        print('Gradient Boosting score:', clf.score(test_x, test_y))
        pred_y = clf.predict(test_x).astype(int)
        print('Gradient Boosting numerical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))
    return

# Time-series models
def time_series(block_num):
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    print('Time-series methods (non-categorical only):')
    autoregression()
    arima()
    sarimax()
    hwes()

def autoregression():
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    model = AutoReg(train_y, lags=10, old_names=False)
    model_fit = model.fit()
    pred_y = model_fit.predict(start=len(train_y), end=len(train_y)+len(test_y)-1)
    print('\tAuto Regression numerical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))

# It combines both Autoregression (AR) and Moving Average (MA) models as well as a differencing pre-processing step of the sequence to make the sequence stationary, called integration (I)
def arima():
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    model = ARIMA(train_y, order=(4,0, 3))
    model_fit = model.fit()
    pred_y = model_fit.predict(start=len(train_y), end=len(train_y)+len(test_y)-1, typ='levels')
    print('\tARIMA numerical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))
    
def sarimax():
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    model = SARIMAX(train_y, order=(0, 1, 1))
    model_fit = model.fit(disp=False)
    pred_y = model_fit.predict(start=len(train_y), end=len(train_y)+len(test_y)-1, typ='levels')
    print('\tSARIMA numerical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))
    
# The Holt Winter’s Exponential Smoothing (HWES) also called the Triple Exponential Smoothing method models the next time step as an exponentially weighted linear function of observations at prior time steps, taking trends and seasonality into account.
def hwes():
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    model = ExponentialSmoothing(train_y, initialization_method="estimated")
    model_fit = model.fit()
    pred_y = model_fit.predict(start=len(train_y), end=len(train_y)+len(test_y)-1)
    print('\tHWES numerical rmse:', np.sqrt(np.sum( (pred_y - test_y)**2 )))

# Simple RNN
def naive_RNN(block_num):
    #loss is computed per prediction, the sum is not calculated
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    print('Naive RNN method:')
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1))
    # model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
    # model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
    if categorical:
        model.add(layers.Dense(train_y.shape[1]))
    else:
        model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mae')
    model.fit(train_x, train_y, epochs=2, verbose=1)
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    if categorical:
        test_y = np.argmax(test_y, axis=1)
        prediction = np.argmax(prediction, axis=1)
        print('categorical rmse:', np.sqrt(np.sum((prediction - test_y)**2 )))
    else:
        print('numerical rmse:', np.sqrt(np.sum( (prediction - test_y)**2 )))
    return
    
if __name__ == '__main__':
    block_num = 196
    print('baseline.py')
    # time_series(block_num)
    # print('#'*60)
    naive_RNN(block_num)
    # print('#'*60)
    # naive_average(block_num)
    # print('#'*60)
    # naive_regression(block_num)
    # print('#'*60)
    # naive_MLP(block_num)
    # print('#'*60)
    # gradient_boost(block_num)
    # print('#'*60)