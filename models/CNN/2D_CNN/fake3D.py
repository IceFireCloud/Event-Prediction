#from sklearn.linear_model import Ridge
import datetime
import keras
from keras.models import Sequential
from keras.layers import *
from keras import optimizers
import numpy as np
import pandas as pd

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# need to change the update function for the weights
# namely, all the weights that are on different channels but same location are the same

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

categorical = False 
num_categories = 32

def one_grid(block_num):
    """Return sum of squared errors for a single grid block"""
    # train and test data
    df = pd.read_csv('../../data/Block_Columns.csv')

    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')

    df.Weekday = df.Date.dt.weekday
    df = df.fillna(value=0)

    train_stop = datetime.datetime(2015, 10, 15)
    val_stop = datetime.datetime(2015, 10, 31) 
    test_stop = datetime.datetime(2016, 11, 30)

    train_mask = df.Date <= train_stop
    val_mask = (df.Date > train_stop)&(df.Date <= val_stop)
    test_mask = (df.Date > val_stop)&(df.Date <= test_stop)

    # make block_num the Y value, then delete it and the date column
    Y = df[str(block_num)]
    del df[str(block_num)]
    del df['Date']
    del df['Training_Split'] # don't need this right now

    signals = df.values
    signals = signals.transpose()
    label = Y.values

    n_signals = signals.shape[0]

    # extract time period
    train_start=0
    train_end=2567
    predict_start= train_end + 7*24
    predict_end = predict_start + 14*24
    input_len = 7*24

    # generate compare signal C
    #compare_signal = [label[0, train_start:train_end+1].mean()]*predict_length
    #compare_signal = np.array(compare_signal)

    # generate time index
    #time_index = range(train_start + 31,train_end + 2)
    #time_index = range(predict_start + 1,predict_end + 2)
    #time_index = np.array(time_index)

    # generate training data and label
    train_x = []
    train_y = []
    # the pointer point to the start of input_length
    pointer = train_start
    label_pointer = pointer + input_len - 1 
    while label_pointer <= train_end:
        train_x_piece = signals[:, pointer:pointer+input_len]
        train_x_piece = np.transpose(train_x_piece)
        train_x.append(train_x_piece)

        train_y_piece = label[label_pointer]
        train_y.append(train_y_piece)

        pointer += 1
        label_pointer += 1

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # generate testing data and label
    test_x = []
    test_y = []
    # pointer
    label_pointer = predict_start
    pointer = label_pointer - input_len + 1
    while label_pointer <= predict_end:
        test_x_piece = signals[:, pointer:pointer + input_len]
        test_x_piece = np.transpose(test_x_piece)
        test_x.append(test_x_piece)

        test_y_piece = label[label_pointer]
        test_y.append(test_y_piece)

        label_pointer += 1
        pointer += 1
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    if categorical:
        train_y = keras.utils.to_categorical(train_y, num_classes=num_categories)
        # don't need this line because Keras never sees the test vector
        #test_y = keras.utils.to_categorical(test_y, num_classes=num_categories)

    # parameter settings
    settings = {}
    #settings['predict_day'] = 3 # predict 3 days ahead
    settings['input_len'] = 7 
    #settings['normalize'] = False 
    settings['pool_size'] = 2 
    settings['model_type'] = 'CNN'
    settings['model_subtype'] = 'CNN'
    settings['label_as_signal'] = True 

    settings['n_epoch'] = 10
    settings['l_r'] = 0.0001
    settings['batch_size'] = 19 
    settings['dropout_rate'] = 0.5
    settings['conv_layers'] = 3 
    settings['num_filters'] = 50 
    settings['first_pool'] = 1  # control pooling of CNN 
    settings['second_pool'] = 1 
    settings['l1_reg'] = 0
    #settings['drop_zeros'] = True 
    #settings['integrate_signals'] = True

    layers = settings['conv_layers']
    dropout_rate = settings['dropout_rate']
    model = Sequential()
    print(train_x.shape)
    # first layer
    model.add(Conv1D(filters=settings['num_filters'],
                    kernel_size=7,
                    strides=1,
                    activation='relu',
                    input_shape=(train_x.shape[1],
                        train_x.shape[2]),
                    kernel_regularizer=regularizers.l1(settings['l1_reg'])))
    if settings['first_pool']:
        model.add(MaxPooling1D(pool_size=2))

    # additional layers as needed
    for i in range(layers - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(Conv1D(filters=(2 * settings['num_filters']),
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        kernel_regularizer=regularizers.l1(settings['l1_reg'])))

    # final layers
    if settings['second_pool']:
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    if categorical:
        model.add(Dense(num_categories, activation='softmax'))
    else:
        model.add(Dense(1, activation='relu'))

    # optimizer
    adam = optimizers.Adam(lr=settings['l_r'])
    if categorical:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mean_squared_error'])
    else:
        model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])
    '''
    # train the model with early stopping when we don't improve
    #callbacks = [EarlyStopping(monitor='loss', patience=1000, verbose=0)]
    model.fit(train_x, train_y, nb_epoch=settings['n_epoch'],
                batch_size=min(settings['batch_size'], train_x.shape[0]),
                verbose=1)
                #validation_data=(test_x, test_y),
                #callbacks=callbacks, verbose=1)

    # prediction and rescale up, if needed
    prediction = model.predict(test_x, batch_size=test_x.shape[0])

    # convert prediction to single vector
    prediction = np.argmax(prediction, axis=1)
    print(sum(prediction))

    results = pd.DataFrame({'truth': test_y,
                            'prediction': prediction})
    filename_s = '../../results/initial_discretized_2_' + str(block_num) + '.csv'
    results.to_csv(filename_s)
    return np.sum( (test_y - prediction)**2 )
    '''

df = pd.read_csv('../../data/Block_Columns.csv')

non_zero_blocks = set(df.columns.values) 
squared_error = 0
squared_error = one_grid(121)
'''for block in range(256):
    if str(block) in non_zero_blocks:
        print('\n', block)
        squared_error += one_grid(block)

N = 256 * 337   # total number of predictions: blocks * hours

print()
print('Sum of squared errors:', squared_error)
print('Mean error:', squared_error / N)
print('RMSE:', np.sqrt(squared_error / N))
'''





