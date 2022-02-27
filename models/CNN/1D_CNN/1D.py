#Last modified: 9/5/20
#Working: yes
# 1D.py

import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import sys
sys.path.insert(0, '/scratch2/xzhou/ep/LA_crime/our_models')
from utils import *
import pandas as pd
import tensorflow.keras
import datetime
import gc
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from data import process_data
from tensorflow.keras.layers import Dense, Dropout

#Limit the GPU usage
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# parameter settings
settings = {}
#settings['predict_day'] = 3 # predict 3 days ahead
#settings['input_len'] = 7
#settings['normalize'] = False
#settings['pool_size'] = 2
#settings['model_type'] = 'CNN'
#settings['model_subtype'] = 'CNN'
#settings['label_as_signal'] = True
settings['n_epoch'] = 60
settings['root'] = '/scratch2/xzhou/ep/LA_crime/results/CNN/'
settings['l_r'] = 0.001
#settings['batch_size'] = 19
settings['dropout_rate'] = 0.5
settings['conv_layers'] = 2
settings['num_filters'] = 50
settings['first_pool'] = 1  # control pooling of CNN
settings['second_pool'] = 1
settings['l1_reg'] = 0.001
#settings['drop_zeros'] = True
#settings['integrate_signals'] = True

categorical = True
gc.collect()
csvfilename = '/scratch2/xzhou/ep/LA_crime/data/Block_Columns.csv'

def one_grid(block_num, save_path):
    """Return sum of squared errors for a single grid block"""
    train_x, train_y, test_x, test_y, average = process_data(block_num, csvfilename, categorical)
    #print(train_x.shape)
    layers = settings['conv_layers']

    model = Sequential()
    
    # first layer
    model.add(Conv1D(filters=settings['num_filters'],
                    kernel_size=7,
                    strides=1,
                    activation='relu',
                    input_shape=(train_x.shape[1], train_x.shape[2]),
                    kernel_regularizer=regularizers.l1(settings['l1_reg'])))
    if settings['first_pool']:
        model.add(MaxPooling1D(pool_size=2))
    # additional layers as needed
    for i in range(layers - 1):
        #model.add(Dropout(settings['dropout_rate']))
        model.add(Conv1D(filters=(2 * settings['num_filters']),
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        kernel_regularizer=regularizers.l1(settings['l1_reg'])))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
    # final layers
    if settings['second_pool']:
        model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    #model.add(Dense(256, activation='relu')) #consider add this layer
    adam = optimizers.Adam(lr=settings['l_r'])
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    if categorical:
        num_categories = int(max([train_x.max(), train_y.max()]))
        model.add(Dense(num_categories, activation='softmax'))#~94
        #model.add(Dense(num_categories, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['mean_squared_error'])
        #optimizer=adam, sgd
    else:
        model.add(Dense(1, activation='softplus'))
        model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

    # train the model with early stopping when we don't improve
    #callbacks = [EarlyStopping(monitor='loss', patience=1000, verbose=0)]
    print(model.summary())
    model.fit(train_x, train_y,
              epochs=settings['n_epoch'],
              batch_size=128,
              #batch_size=min(settings['batch_size'], train_x.shape[0]),
              verbose=1)
                #validation_data=(train_x, train_y))#,
                #callbacks=callbacks, verbose=1)
    score = model.evaluate(train_x, train_y, batch_size=128)
    print('training rmse from tensorflow.keras: ', np.sqrt(score[1]))
    score = model.evaluate(test_x, test_y, batch_size=128)
    print('testing rmse from tensorflow.keras:  ', np.sqrt(score[1]))
    #...
    # prediction and rescale up, if needed
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    # convert prediction to single vector
    if categorical:
        prediction = np.argmax(prediction, axis=1)
        test_y = np.argmax(test_y, axis=1)
        results = pd.DataFrame({'truth': test_y, 'prediction': prediction})
    else:
        results = pd.DataFrame({'truth': test_y, 'prediction': prediction.flatten()})
    print('(sum of pred, test, lenth): ', sum(prediction), sum(test_y), len(test_y))
    filename_s = save_path + '/block_' + str(block_num)
    results.to_csv(filename_s + '.csv')
    draw(filename_s, average, True, True)
    with open(save_path + '/structure.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    trainning_sse = np.sum((np.argmax(model.predict(train_x), axis=1)-np.argmax(train_y, axis=1))**2)
    return np.sum( (test_y - prediction)**2 ), trainning_sse, model

filename = str(datetime.datetime.now())[:13].replace(' ', '-')
save_path = settings['root']+filename
if not os.path.isdir(save_path):
    os.mkdir(save_path)

df = pd.read_csv(csvfilename)
non_zero_blocks = set(df.columns.values)
squared_error = 0
#squared_error = one_grid(121, save_path)
print('\n\nnow processing block 196\n\n')
squared_error = one_grid(196, save_path)
'''
for block in range(256):
    if str(block) in non_zero_blocks:
        print('\n', block)
        squared_error += one_grid(block)
'''
#N = 256 * 337   # total number of predictions: blocks * hours

print()
print('Sum of squared errors: (prediction, trainning_sse)', squared_error)
#print('Mean error:', squared_error / N)
#print('RMSE:', np.sqrt(squared_error / N))

with open(save_path+'/parameters.txt', 'w') as f:
    f.write(json.dumps(settings))

# save model for later use
_, _, model = squared_error
model.save(save_path+'/model.h5')