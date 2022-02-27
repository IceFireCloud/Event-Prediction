import os
import sys
#sys.stdout = open(os.devnull, 'w')
#sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
import scipy.io
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from time import sleep
import json

from base_model import BaseModel

class Model(BaseModel):
    def run_model(self):
        # output the logs to a file
        try:
            sys.stdout = open(self.result_dir + 'logs/' + self.just_name + '.txt', 'w+')
        except:
            os.makedirs(self.result_dir + 'logs/')
            sys.stdout = open(self.result_dir + 'logs/' + self.just_name + '.txt', 'w+')

        ## generate new model
        layers = self.settings['conv_layers']
        dropout_rate = self.settings['dropout_rate']
        model = Sequential()

        # first layer
        #model.add(Dropout(rate=dropout_rate,
        #                        input_shape=(self.train_x.shape[1], self.train_x.shape[2])))
        model.add(Conv1D(filters=self.settings['num_filters'],
                        kernel_size=7,
                        strides=1,
                        activation='relu',
                        input_shape=(self.train_x.shape[1],
                            self.train_x.shape[2]),
                        kernel_regularizer=regularizers.l1(self.settings['l1_reg'])))
        if self.settings['first_pool']:
            model.add(MaxPooling1D(pool_size=2))

        # additional layers as needed
        for i in range(layers - 1):
            model.add(Dropout(rate=dropout_rate))
            model.add(Conv1D(filters=(2 * self.settings['num_filters']),
                            kernel_size=3,
                            strides=1,
                            activation='relu',
                            kernel_regularizer=regularizers.l1(self.settings['l1_reg'])))

        # final layers
        if self.settings['second_pool']:
            model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='relu'))

        # optimizer
        adam = Adam(lr=self.settings['l_r'])
        model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

        # time
        start_time = time.time()

        # normalize the labels output to fit between zero and one
        if self.settings['normalize']:
            hi = np.amax(self.train_y)
            if hi != 0:
                self.train_y = self.train_y / hi

        # train the model with early stopping when we don't improve
        callbacks = [EarlyStopping(monitor='loss', patience=1000, verbose=0)]
        model.fit(self.train_x, self.train_y, nb_epoch=self.settings['n_epoch'],
                    batch_size=min(self.settings['batch_size'], self.train_x.shape[0]),
                    validation_data=(self.test_x, self.test_y),
                    callbacks=callbacks, verbose=1)

        end_time = time.time()
        self.train_time = end_time - start_time

        print('training time:' + str(self.train_time))

        # prediction and rescale up, if needed
        prediction = model.predict(self.test_x, batch_size=self.test_x.shape[0])
        #prediction = model.predict(self.train_x, batch_size=self.train_x.shape[0])
        if self.settings['normalize']:
            if hi != 0:
                prediction = prediction * hi

        # reshape the prediction to 1-D array
        self.P = np.sum(prediction, axis=1)

        # change back the output
        sys.stdout = sys.__stdout__
        
        # return the model so we can use information about it
        #model.save('/home/schin/Mercury/Chin/new_frame/CNN/models/no_label_model.h5')
        self.keras_model = model
        return(model)



