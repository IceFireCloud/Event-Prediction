import numpy as np
import pandas as pd
import keras
import datetime
import gc
from keras.models import Sequential
from keras.layers import *
from keras import optimizers
from keras.optimizers import SGD

# parameter settings
settings = {}
#settings['predict_day'] = 3 # predict 3 days ahead
settings['input_len'] = 7 
#settings['normalize'] = False 
settings['pool_size'] = 2 
settings['model_type'] = 'CNN'
settings['model_subtype'] = 'CNN'
settings['label_as_signal'] = True 

settings['n_epoch'] = 60
settings['l_r'] = 0.0001
settings['batch_size'] = 19 
settings['dropout_rate'] = 0.5
settings['conv_layers'] = 2
settings['num_filters'] = 50 
settings['first_pool'] = 1  # control pooling of CNN 
settings['second_pool'] = 1 
settings['l1_reg'] = 0
#settings['drop_zeros'] = True 
#settings['integrate_signals'] = True

categorical = False 
num_categories = 32
gc.collect()




def one_grid(block_num):
    """Return sum of squared errors for a single grid block"""
    #2401*168*112 => 2401*168*256 => 2401*16*16*168
    # train and test data
    df = pd.read_csv('../../data/Block_Columns.csv')

    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')

    df.Weekday = df.Date.dt.weekday
    df = df.fillna(value=0)
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

    # extract time period
    train_start=0
    train_end=2567
    predict_start= train_end + 7*24
    predict_end = predict_start + 14*24
    input_len = 7*24
    predict_hours = 1*24

    # generate training data and label
    train_x = []
    train_y = []
    # the pointer point to the start of input_length
    pointer = train_start
    label_pointer = pointer + input_len - 1 
    while label_pointer <= train_end:
        train_x_piece = signals[pointer:pointer+input_len, :, :]
        train_x.append(train_x_piece.transpose())

        train_y_piece = label[label_pointer+predict_hours]
        train_y.append(train_y_piece)

        pointer += 1
        label_pointer += 1

    train_x = np.array(train_x)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3], 1)
    train_y = np.array(train_y)

    # generate testing data and label
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
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], test_x.shape[3], 1)
    print(test_x.shape, train_x.shape)
    
    if categorical:
        train_y = keras.utils.to_categorical(train_y, num_classes=num_categories)
        # don't need this line because Keras never sees the test vector
        #test_y = keras.utils.to_categorical(test_y, num_classes=num_categories)

    layers = settings['conv_layers']
    dropout_rate = settings['dropout_rate']
    model = Sequential()
    # first layer
    #Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    
    #Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    model.add(Conv3D(filters=settings['num_filters'],
                                kernel_size=7,
                                strides=1,
                                activation='relu',
                                input_shape=(train_x.shape[1],train_x.shape[2], train_x.shape[3], 1),
                                kernel_regularizer=regularizers.l1(settings['l1_reg'])))
    if settings['first_pool']:
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # additional layers as needed
    for i in range(layers - 1):
        model.add(Dropout(rate=dropout_rate))#probably after the layer?
        model.add(Conv3D(filters=(2 * settings['num_filters']),
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        kernel_regularizer=regularizers.l1(settings['l1_reg'])))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu')) #consider add this layer
    if categorical:
        model.add(Dense(num_categories, activation='softmax'))
    else:
        #model.add(Dense(1, activation='relu')) #sigmoid
        model.add(Dense(1, activation='sigmoid')) #sigmoid

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

    # prediction and rescale up, if needed
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    #print(prediction)
    # convert prediction to single vector
    #print(prediction)
    #print(prediction.flatten())
    #prediction = prediction[np.argmax(prediction, axis=1)]
   # print(test_y)
    print(sum(prediction), sum(test_y), len(test_y))
    #print(prediction)

    results = pd.DataFrame({'truth': test_y,
                            'prediction': prediction.flatten()})
    filename_s = '../../results/3D/initial_discretized_2_' + str(block_num) + '.csv'
    results.to_csv(filename_s)

    return np.sum( (test_y - prediction)**2 )

df = pd.read_csv('../../data/Block_Columns.csv')

non_zero_blocks = set(df.columns.values) 
squared_error = 0
#squared_error = one_grid(121)
#'''
for block in range(256):
    if str(block) in non_zero_blocks:
        print('\n', block)
        squared_error += one_grid(block)
#'''
N = 256 * 337   # total number of predictions: blocks * hours

print()
print('Sum of squared errors:', squared_error)
print('Mean error:', squared_error / N)
print('RMSE:', np.sqrt(squared_error / N))