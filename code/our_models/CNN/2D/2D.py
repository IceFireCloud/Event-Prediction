#Modified for adapting the new environment. 18/11/20
#Last modified: 18/12/11
#Working? 

import numpy as np
import pandas as pd
import keras
import datetime
import gc
import os
import json
from keras.models import Sequential
from keras.layers import *
from keras import optimizers
from keras.optimizers import SGD

#Limit the GPU usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# parameter settings
settings = {}
#settings['predict_day'] = 3 # predict 3 days ahead
#settings['input_len'] = 7 
#settings['normalize'] = False 
#settings['pool_size'] = 2 
#settings['model_type'] = 'CNN'
#settings['model_subtype'] = 'CNN'
#settings['label_as_signal'] = True 
settings['n_epoch'] = 100
settings['root'] = '/data/schin/Work/LA_crime/results/CNN/'
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


from keras.layers import Dense, Dropout



categorical = True
gc.collect()
csvfilename = '/data/schin/Work/LA_crime/data/Block_Columns.csv'

def process_data(block_num):
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

def one_grid(block_num, save_path):
    """Return sum of squared errors for a single grid block"""
    train_x, train_y, test_x, test_y, average = process_data(block_num)
    
    #print(train_x.shape)
    layers = settings['conv_layers']
    
    model = Sequential()
    # first layer
    model.add(Conv2D(filters=settings['num_filters'],
                    kernel_size=7,
                    strides=1,
                    activation='relu',
                    input_shape=(train_x.shape[1],train_x.shape[2], train_x.shape[3]),
                    kernel_regularizer=regularizers.l1(settings['l1_reg'])))
    if settings['first_pool']:
        model.add(MaxPooling2D(pool_size=(2,2)))
    
    # additional layers as needed
    for i in range(layers - 1):
        #model.add(Dropout(settings['dropout_rate']))
        model.add(Conv2D(filters=(2 * settings['num_filters']),
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        kernel_regularizer=regularizers.l1(settings['l1_reg'])))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
    # final layers
    if settings['second_pool']:
        model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    #model.add(Dense(256, activation='relu')) #consider add this layer
    adam = optimizers.Adam(lr=settings['l_r'])
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    if categorical:
        num_categories = int(max([train_x.max(), train_y.max()]))
        model.add(Dense(num_categories, activation='softmax'))#~94
        #model.add(Dense(num_categories, activation='sigmoid'))
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['mean_squared_error'])
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mean_squared_error'])
        #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mean_squared_error'])
    else:
        model.add(Dense(1, activation='softplus'))
        model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

    # train the model with early stopping when we don't improve
    #callbacks = [EarlyStopping(monitor='loss', patience=1000, verbose=0)]
    model.fit(train_x, train_y,
              epochs=settings['n_epoch'],
              batch_size=128,
              #batch_size=min(settings['batch_size'], train_x.shape[0]),
              verbose=1)
                #validation_data=(train_x, train_y))#,
                #callbacks=callbacks, verbose=1)
    score = model.evaluate(train_x, train_y, batch_size=128)
    print('training rmse from keras: ', np.sqrt(score[1]))
    score = model.evaluate(test_x, test_y, batch_size=128)
    print('test rmse from keras: ', np.sqrt(score[1]))
    #...
    # prediction and rescale up, if needed
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    prediction_train = model.predict(train_x)
    # convert prediction to single vector
    #print(prediction)
    #print(prediction.flatten())
    #prediction = prediction[np.argmax(prediction, axis=1)]
    #print(test_y)
    #print(np.argmax(prediction, axis=1))
    
    if categorical:
        prediction_train = np.argmax(prediction_train, axis=1)
        prediction = np.argmax(prediction, axis=1)
        test_y = np.argmax(test_y, axis=1)
        train_y = np.argmax(train_y, axis=1)
        results = pd.DataFrame({'truth': test_y,
                                'prediction': prediction})
        results2 = pd.DataFrame({'truth': train_y,
                                'prediction': prediction_train})
    else:
        results = pd.DataFrame({'truth': test_y,
                                'prediction': prediction.flatten()})
        results2 = pd.DataFrame({'truth': train_y,
                                'prediction': prediction_train.flatten()})
    print('(sum of pred, test, lenth): ', sum(prediction), sum(test_y), len(test_y))
    filename_1 = save_path + '/block_' + str(block_num) + '.csv'
    filename_2 = save_path + '/block_' + str(block_num) + '_train.csv'
    results.to_csv(filename_1)
    results2.to_csv(filename_2)
    with open(save_path + '/structure.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    return np.sum( (test_y - prediction)**2 ), np.sum( (average - test_y)**2 ), np.sum( (np.array(int(average+0.5)) - test_y)**2 ), np.sum((prediction_train-train_y)**2), model

filename = str(datetime.datetime.now())[:13].replace(' ', '-')
save_path = settings['root']+filename
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
df = pd.read_csv(csvfilename)
non_zero_blocks = set(df.columns.values) 
squared_error = 0
squared_error = one_grid(121, save_path)
'''
for block in range(256):
    if str(block) in non_zero_blocks:
        print('\n', block)
        squared_error += one_grid(block)
'''
#N = 256 * 337   # total number of predictions: blocks * hours

print()
print('Sum of squared errors: (prediction, average, categorical-average, trainning_sse)', squared_error)
#print('Mean error:', squared_error / N)
#print('RMSE:', np.sqrt(squared_error / N))

with open(save_path+'/parameters.txt', 'w') as f:
    f.write(json.dumps(settings))
    
# save model for later use
_, _, _, _, model = squared_error
model.save(save_path+'/model.h5')

'''
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
'''