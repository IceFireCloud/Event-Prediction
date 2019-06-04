# A neural network that learns to combine the outputs of other trained networks
# somehow similar to adaboost


import sys
sys.path.insert(0, '/data/schin/Work/LA_crime/our_models')
from csv2fig import *
import numpy as np
import pandas as pd
import datetime
import gc
import os
import json
import sys
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import *
from keras import optimizers
from keras.optimizers import SGD

gc.collect()
csvfilename = '/data/schin/Work/LA_crime/data/Block_Columns.csv'

settings = {}
settings['n_epoch'] = 300
settings['root'] = '/data/schin/Work/LA_crime/results/Combiner/'

categorical = True
train_csv = True

def process_data_1d(block_num):
    # train and test data
    df = pd.read_csv(csvfilename)
    df = df.fillna(value=0)
    #2401*168*112 => 2401*168*256 => 2401*16*16*168
    df2 = pd.DataFrame()
    for i in range(256):
        df2[str(i)] = np.zeros(4416,dtype=int)
    for i in range(256):
        if str(i) in df:
            df2[str(i)] = df[str(i)]
    df = df2
    # make block_num the Y value, then delete it and the date column
    Y = df[str(block_num)]
    signals = df.values
    '''
    temp = np.zeros((len(signals), 16, 16))
    for i in range(len(signals)):
        temp[i] = signals[i].reshape(16, 16)
    #signals = signals.transpose()
    signals = temp
    '''
    #print(signals.shape)
    label = Y.values

    # generate data and label for training and testing
    train_start = 0
    train_end = 2567
    predict_start= train_end + 7*24
    predict_end = predict_start + 14*24
    input_len = 7*24
    predict_hours = 1*24
    #single output

    train_x = []
    train_y = []
    pointer = train_start
    label_pointer = pointer + input_len - 1
    while label_pointer <= train_end:
        #train_x_piece = signals[pointer:pointer+input_len, :, :]
        train_x_piece = signals[pointer:pointer+input_len, :]
        train_x.append(train_x_piece.transpose())

        train_y_piece = label[label_pointer+predict_hours] #here delayed 24 hours
        train_y.append(train_y_piece)

        pointer += 1
        label_pointer += 1

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x = []
    test_y = []
    # pointer
    label_pointer = predict_start
    pointer = label_pointer - input_len + 1
    while label_pointer <= predict_end:
        #test_x_piece = signals[pointer:pointer + input_len, :, :]
        test_x_piece = signals[pointer:pointer + input_len, :]
        test_x.append(test_x_piece.transpose())

        test_y_piece = label[label_pointer+predict_hours]
        test_y.append(test_y_piece)

        label_pointer += 1
        pointer += 1
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    #print(test_x.shape, train_x.shape)
    average = np.mean(train_y)

    if categorical:
        num_categories = int(signals.max())
        #print(num_categories)
        train_y = keras.utils.to_categorical(train_y, num_classes=num_categories)
        # don't need this line because Keras never sees the test vector
        test_y = keras.utils.to_categorical(test_y, num_classes=num_categories)
    return train_x, train_y, test_x, test_y, average

def process_data_mlp(block_num):
    # train and test data
    df = pd.read_csv(csvfilename)
    #df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    #df.Weekday = df.Date.dt.weekday
    df = df.fillna(value=0)
    #2401*168*112 => 2401*168*256 => 2401*16*16*168
    df2 = pd.DataFrame()
    for i in range(256):
        df2[str(i)] = np.zeros(4416,dtype=int)
    for i in range(256):
        if str(i) in df:
            df2[str(i)] = df[str(i)]
    df = df2
    # make block_num the Y value, then delete it and the date column
    Y = df[str(block_num)]
    #del df['Date']
    #del df['Training_Split'] # don't need this right now
    signals = df.values
    temp = np.zeros((len(signals), 16, 16))
    for i in range(len(signals)):
        temp[i] = signals[i].reshape(16, 16)
    #signals = signals.transpose()
    signals = temp
    #print(signals.shape)
    label = Y.values

    # generate data and label for training and testing
    train_start = 0
    train_end = 2567
    predict_start= train_end + 7*24
    predict_end = predict_start + 14*24
    input_len = 7*24
    predict_hours = 1*24
    #single output

    train_x = []
    train_y = []
    pointer = train_start
    label_pointer = pointer + input_len - 1
    while label_pointer <= train_end:
        train_x_piece = signals[pointer:pointer+input_len, :, :]
        train_x.append(train_x_piece.transpose())

        train_y_piece = label[label_pointer+predict_hours] #here delayed 24 hours
        train_y.append(train_y_piece)

        pointer += 1
        label_pointer += 1

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x = []
    test_y = []
    # pointer
    label_pointer = predict_start
    pointer = label_pointer - input_len + 1
    while label_pointer <= predict_end:
        test_x_piece = signals[pointer:pointer + input_len, :, :]
        test_x.append(test_x_piece.transpose())

        test_y_piece = label[label_pointer+predict_hours]
        test_y.append(test_y_piece)

        label_pointer += 1
        pointer += 1
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    #print(test_x.shape, train_x.shape)
    #Compare signals
    average = np.mean(train_y)
    if categorical:
        num_categories = int(signals.max())
        #print(num_categories)
        train_y = keras.utils.to_categorical(train_y, num_classes=num_categories)
        # don't need this line because Keras never sees the test vector
        test_y = keras.utils.to_categorical(test_y, num_classes=num_categories)
    temp = np.zeros([train_x.shape[0], train_x.shape[1]*train_x.shape[2]*train_x.shape[3]])
    for i in range(train_x.shape[0]):
        temp[i] = train_x[i].flatten()
    #signals = signals.transpose()
    train_x = temp
    temp = np.zeros([test_x.shape[0], test_x.shape[1]*test_x.shape[2]*test_x.shape[3]])
    for i in range(test_x.shape[0]):
        temp[i] = test_x[i].flatten()
    #signals = signals.transpose()
    test_x = temp
    return train_x, train_y, test_x, test_y, average

############load_model########################
CNN_model1 = load_model('/data/schin/Work/LA_crime/results/CNN/2018-12-14-13/model.h5')
CNN_model2 = load_model('/data/schin/Work/LA_crime/results/CNN/2018-12-13-16/model.h5')
MLP_model = load_model('/data/schin/Work/LA_crime/results/MLP/2019-02-02-06/model.h5')

############prepare data######################
train_x_1d, train_y_1d, test_x_1d, test_y_1d, average_1d = process_data_1d(block_num=121)
train_x_mlp, train_y_mlp, test_x_mlp, test_y_mlp, average_mlp = process_data_mlp(block_num=121)

train_x1 = CNN_model1.predict(train_x_1d)
train_x2 = CNN_model2.predict(train_x_1d)
train_x3 = MLP_model.predict(train_x_mlp)
#train_x = np.concatenate([train_x1, train_x2, train_x3, (average_1d-train_x3+train_x3)], axis=1)
train_x = np.concatenate([train_x1, train_x2, train_x3], axis=1)
train_y = train_y_1d

test_x1 = CNN_model1.predict(test_x_1d)
test_x2 = CNN_model2.predict(test_x_1d)
test_x3 = MLP_model.predict(test_x_mlp)
#test_x = np.concatenate([test_x1, test_x2, test_x3, (average_1d-test_x3+test_x3)], axis=1)
test_x = np.concatenate([test_x1, test_x2, test_x3], axis=1)
test_y = test_y_1d

average = average_1d
with open('average', 'w') as f:
    f.write(str(average))

############Build model#######################
model = Sequential()
model.add(Dense(32, input_shape=(train_x.shape[1],), activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
if categorical:
    model.add(Dense(train_y.shape[1], activation='softmax'))#~94
    #model.add(Dense(train_y.shape[1], activation='sigmoid'))#~
    model.compile(loss='categorical_crossentropy',
                  #optimizer='rmsprop',
                  #optimizer='adam',
                  optimizer=sgd,
                  metrics=['mean_squared_error'])
else:
    model.add(Dense(1, activation='softplus'))
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mean_squared_error'])

############training##########################
model.fit(train_x, train_y,
          epochs=settings['n_epoch'],
          batch_size=128)

############results###########################
score = model.evaluate(test_x, test_y, batch_size=128)
prediction = model.predict(test_x, batch_size=test_x.shape[0])

filename = str(datetime.datetime.now())[:13].replace(' ', '-')
save_path = settings['root']+filename
if not os.path.isdir(save_path):
    os.mkdir(save_path)

df = pd.read_csv(csvfilename)
non_zero_blocks = set(df.columns.values)

filename_s = save_path + '/block_' + str(121)
if categorical:
    prediction = np.argmax(prediction, axis=1)
    test_y = np.argmax(test_y, axis=1)
    results = pd.DataFrame({'truth': test_y,
                            'prediction': prediction})

else:
    results = pd.DataFrame({'truth': test_y,
                            'prediction': prediction.flatten()})

if train_csv:
    prediction_train = model.predict(train_x)
    prediction_train = np.argmax(prediction_train, axis=1)
    train_y = np.argmax(train_y, axis=1)
    results_train = pd.DataFrame({'truth': train_y, 'prediction': prediction_train})
    results_train.to_csv(filename_s + '_train.csv')

print('(sum of pred, test, lenth): ', sum(prediction), sum(test_y), len(test_y))
results.to_csv(filename_s+'.csv')
with open(filename_s + '_model.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

results =  np.sum( (test_y - prediction)**2 ), np.sum( (average - test_y)**2 ), np.sum( (np.array(int(average+0.5)) - test_y)**2 ), model

'''
for block in range(256):
    if str(block) in non_zero_blocks:
        print('\n', block)
        squared_error += one_grid(block)
'''
#N = 256 * 337   # total number of predictions: blocks * hours

print()
print('Sum of squared errors: (prediction, average, categorical-average)', results)
#print('Mean error:', squared_error / N)
#print('RMSE:', np.sqrt(squared_error / N))

with open(save_path+'/parameters.txt', 'w') as f:
    f.write(json.dumps(settings))

# save model for later use
_, _, _, model = results
model.save(save_path+'/model.h5')

#draw graph
draw(filename_s, True)
draw(filename_s + '_train', True)
