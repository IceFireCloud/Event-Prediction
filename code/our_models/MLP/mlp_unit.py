import numpy as np
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

#Limit the GPU usage
import tensorflow as tf
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
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
settings['n_epoch'] = 10
settings['root'] = '/data/schin/Work/LA_crime/results/MLP/'
#settings['l_r'] = 0.0001
#settings['batch_size'] = 19
#settings['dropout_rate'] = 0.5
#settings['conv_layers'] = 2
#settings['num_filters'] = 50
#settings['first_pool'] = 1  # control pooling of CNN
#settings['second_pool'] = 1
settings['l1_reg'] = 0.00
#settings['drop_zeros'] = True
#settings['integrate_signals'] = True


from tensorflow.keras.layers import Dense, Dropout



categorical = True
gc.collect()
csvfilename = '/data/schin/Work/LA_crime/data/Block_Columns.csv'
num_categories = 1

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
    print(signals.shape)
    temp = np.zeros((len(signals), 16, 16))
    for i in range(len(signals)):
        temp[i] = signals[i].reshape(16, 16)
    #signals = signals.transpose()
    signals = temp
    #print(signals.shape)
    label = Y.values

    # generate data and label for training and testing
    train_start = 0
    train_end = 3850
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
        train_y = tensorflow.keras.utils.to_categorical(train_y, num_classes=num_categories)
        # don't need this line because tensorflow.keras never sees the test vector
        test_y = tensorflow.keras.utils.to_categorical(test_y, num_classes=num_categories)

    return train_x, train_y, test_x, test_y, average

def one_grid(block_num, save_path):
    """Return sum of squared errors for a single grid block"""
    train_x, train_y, test_x, test_y, average = process_data(block_num)

    model = Sequential()
    model.add(Dense(32, input_shape=(train_x.shape[1]*train_x.shape[2]*train_x.shape[3],), activation='tanh'))
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

    model.fit(train_x, train_y,
              epochs=settings['n_epoch'],
              batch_size=128)
    score = model.evaluate(train_x, train_y, batch_size=128)
    print('training rmse from tensorflow.keras: ', np.sqrt(score[1]))
    score = model.evaluate(test_x, test_y, batch_size=128)
    print('testing rmse from tensorflow.keras:  ', np.sqrt(score[1]))
    '''
    layers = settings['conv_layers']
    dropout_rate = settings['dropout_rate']
    model = Sequential()
    # first layer
    model.add(Conv2D(filters=settings['num_filters'],
                    kernel_size=7,
                    strides=1,
                    activation='relu',
                    input_shape=(train_x.shape[1],train_x.shape[2], train_x.shape[3]),
                    kernel_regularizer=regularizers.l1(settings['l1_reg'])))
    if settings['first_pool']:
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # additional layers as needed
    for i in range(layers - 1):
        model.add(Dropout(rate=dropout_rate))#probably after the layer?
        model.add(Conv2D(filters=(2 * settings['num_filters']),
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        kernel_regularizer=regularizers.l1(settings['l1_reg'])))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu')) #consider add this layer
    if categorical:
        model.add(Dense(num_categories, activation='softmax'))
    else:
        model.add(Dense(1, activation='relu')) #sigmoid

    # optimizer
    adam = optimizers.Adam(lr=settings['l_r'])
    if categorical:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mean_squared_error'])
    else:
        model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

    # train the model with early stopping when we don't improve
    #callbacks = [EarlyStopping(monitor='loss', patience=1000, verbose=0)]
    model.fit(train_x, train_y, nb_epoch=settings['n_epoch'],
                batch_size=min(settings['batch_size'], train_x.shape[0]),
                verbose=1)
                #validation_data=(train_x, train_y))#,
                #callbacks=callbacks, verbose=1)
    '''
    # prediction and rescale up, if needed
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    # convert prediction to single vector
    #print(prediction)
    #print(prediction.flatten())
    #prediction = prediction[np.argmax(prediction, axis=1)]
   # print(test_y)
    #print(prediction)
    #print(np.argmax(prediction, axis=1))

    filename_s = save_path + '/block_' + str(block_num)
    if categorical:
        prediction = np.argmax(prediction, axis=1)
        test_y = np.argmax(test_y, axis=1)
        results = pd.DataFrame({'truth': test_y,
                                'prediction': prediction})
    else:
        results = pd.DataFrame({'truth': test_y,
                                'prediction': prediction.flatten()})
    print('(sum of pred, test, lenth): ', sum(prediction), sum(test_y), len(test_y))
    results.to_csv(filename_s+'.csv')
    with open(filename_s + '_model.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    return np.sum( (test_y - prediction)**2 ), np.sum( (average - test_y)**2 ), np.sum( (np.array(int(average+0.5)) - test_y)**2 ), model

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
print('Sum of squared errors: (prediction, average, categorical-average)', squared_error)
#print('Mean error:', squared_error / N)
#print('RMSE:', np.sqrt(squared_error / N))

with open(save_path+'/parameters.txt', 'w') as f:
    f.write(json.dumps(settings))

# save model for later use
_, _, _, model = squared_error
model.save(save_path+'/model.h5')
'''
from tensorflow.keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
'''
