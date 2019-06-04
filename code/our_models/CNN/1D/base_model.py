import os
import sys
#sys.stdout = open(os.devnull, 'w')
#sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
import scipy.io
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json

class BaseModel(object):
    def __init__(self, data_dir, file_name, result_dir, case, settings):
        self.data_dir = data_dir
        self.file_name = file_name
        self.just_name = file_name[:-4]
        self.result_dir = result_dir + 'case' + str(case[4]) + '/'
        self.case = case
        self.settings = settings
        # the label is 1-D numpy array, the data depend on the requirement of the network
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        # truth , 1-D numpy array
        self.T = []
        # prediction, 1-D numpy array
        self.P = []
        # compare, 1-D numpy array
        self.C = []
        # fusion, 1-D numpy array, optional
        self.F = []
        # the date of prediction
        self.time_index = []
        # MSEs and the ratios
        self.MSE_ratio = 0   # prediction divided by compare
        # training time of this model
        self.train_time = 0

    def data_processing(self):
        file_dir = self.data_dir + self.file_name
        # load the data
        load_data = scipy.io.loadmat(file_dir)
        signals = load_data['X']
        #signals = load_data['y']
        signals = np.transpose(signals)
        label = load_data['y']
        # X and y here are both 2-D numpy array
        # concatenate label signal to data signals
        if self.settings['label_as_signal']:
            signals = np.concatenate((signals, label), axis=0)
        n_signals = signals.shape[0]

        # extract time period
        train_start = self.case[0]
        train_end = self.case[1]
        predict_start = self.case[2]
        predict_end = self.case[3]
        case_index = self.case[4]
        train_length = train_end - train_start + 1
        predict_length = predict_end - predict_start + 1

        # drop zero channels if needed
        if self.settings['drop_zeros']:
            to_delete = []
            for i in range(signals.shape[0]):
                if np.max(signals[i, :train_end]) == 0:
                    to_delete.append(i)
            signals = np.delete(signals, to_delete, axis=0)

        # add integrated channels if needed
        try:
            if self.settings['integrate_signals']:
                integrated = np.cumsum(signals, axis=0)
                signals = np.concatenate(signals, integrated, axis=1)
        except: 
            pass

        # generate compare signal C
        compare_signal = [label[0, train_start:train_end+1].mean()]*predict_length
        compare_signal = np.array(compare_signal)

        # generate time index
        #time_index = range(train_start + 31,train_end + 2)
        time_index = range(predict_start + 1,predict_end + 2)
        time_index = np.array(time_index)

        # generate training data and label
        train_x = []
        train_y = []
        # the pointer point to the start of input_length
        pointer = train_start
        label_pointer = pointer + self.settings['input_len'] - 1 + self.settings['predict_day']
        while label_pointer <= train_end:
            train_x_piece = signals[:, pointer:pointer+self.settings['input_len']]
            train_x_piece = np.transpose(train_x_piece)
            train_x.append(train_x_piece)

            train_y_piece = label[0,label_pointer]
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
        pointer = label_pointer - self.settings['input_len'] - self.settings['predict_day'] + 1
        while label_pointer <= predict_end:
            test_x_piece = signals[:, pointer:pointer + self.settings['input_len']]
            test_x_piece = np.transpose(test_x_piece)
            test_x.append(test_x_piece)

            test_y_piece = label[0, label_pointer]
            test_y.append(test_y_piece)

            label_pointer += 1
            pointer += 1
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        # dimension check
        if train_x.shape[0] == train_y.shape[0]:
            self.train_x = train_x
            self.train_y = train_y
        else:
            print('training set dimension mismatch')
            print(train_x.shape)
            print(train_y.shape)
        if test_x.shape[0] == test_y.shape[0] == compare_signal.shape[0] == time_index.shape[0]:
            self.test_x = test_x
            self.test_y = test_y

            #self.T = train_y
            #self.C = train_y 
            self.T = test_y
            self.C = compare_signal
            self.time_index = time_index
        else:
            print('testing set dimension mismatch')
            print('test_x' + str(test_x.shape))
            print('test_y' + str(test_y.shape))
            print('compare' + str(compare_signal.shape))
            print('time_index' + str(time_index.shape))

    def run_model(self):
        """Implement in child class"""
        pass

    def output_result(self, signals, sub_folder):
        # time_index : 1-D array
        # truth : 1-D array
        # signals : dictionary, 'name' : 1-D array
        save_dir = self.result_dir + sub_folder + '/'
        try:
            os.makedirs(save_dir)
        except:
            pass

        sys.stdout = sys.__stdout__
        # compute MSE against the truth
        
        MSE_dic = {}
        for k in signals.keys():
            signal = signals[k]
            MSE = ((signal - self.T) ** 2).sum() / len(self.T)
            MSE_dic[k] = MSE
            if k == 'compare':
                self.C_MSE = MSE
            elif k == 'prediction':
                self.P_MSE = MSE

        if self.C_MSE != 0:
            self.MSE_ratio = self.P_MSE / self.C_MSE 
        else:
            self.MSE_ratio = float('nan')
        
        # draw the graph
        #signals['compare'] = signals['compare'] * 0

        fig = plt.figure()
        line_truth = plt.plot(self.time_index, self.T, label='truth')
       
        for k in signals.keys():
            legend = k + ' MSE=' + str(MSE_dic[k])
            line = plt.plot(self.time_index, signals[k], label=legend)
        plt.legend()
        
        try:
            plt.savefig(save_dir + 'Image/' + self.just_name + '.png')
        except:
            os.mkdir(save_dir + 'Image/')
            plt.savefig(save_dir + 'Image/' + self.just_name + '.png')
        plt.close(fig)

        # save .mat
        mat_save_dic = signals
        mat_save_dic['truth'] = self.T
        mat_save_dic['time_index'] = self.time_index
        try:
            scipy.io.savemat(save_dir + 'MAT/' + self.just_name + '.mat', mat_save_dic)
        except:
            os.mkdir(save_dir + 'MAT/')
            scipy.io.savemat(save_dir + 'MAT/' + self.just_name + '.mat', mat_save_dic)
            
        # save CSV
        data = scipy.io.loadmat(save_dir+'MAT/'+self.just_name+'.mat')
        info = [data['time_index'][0], data['prediction'][0]]
        info = np.array(info).T
        
        try:
            np.savetxt(save_dir+'CSV/'+self.just_name+'.csv', info, delimiter=',', fmt='%s,%.4f')
        except:
            os.mkdir(save_dir + 'CSV/')
            np.savetxt(save_dir+'CSV/'+self.just_name+'.csv', info, delimiter=',', fmt='%s,%.4f')

    def save_settings(self, model_index, experiment_index):
        """save data to central CSV, Model_Runs.csv"""
        # process data we need
        config = json.loads(self.keras_model.to_json())['config'] # extract all architecture info from model
        layers = []
        for d in config:
            layers.append(d['class_name']) 

        # assemble all data
        data = [model_index,
                experiment_index,
                datetime.now(),
                self.settings['comment'][0],
                self.settings['comment'][1],
                self.settings['comment'][2],
                self.file_name,
                self.case,
                self.case[-1],
                self.C_MSE,
                self.P_MSE,
                self.MSE_ratio,
                self.settings['model_type'], 
                self.settings['model_subtype'],
                self.data_dir,
                self.result_dir,
                self.settings['predict_day'],
                self.settings['label_as_signal'],
                self.settings['input_len'],
                self.settings['n_epoch'],
                self.settings['batch_size'],
                self.settings['l_r'],
                self.train_time,
                layers,
                self.keras_model.count_params(),
                config
                ]
        data = [str(x).replace(' ', '') for x in data]  # convert to string and remove all spaces

        # next, open file and append our data, formatted with space separation
        with open('Model_Results.csv', 'a') as f:
            f.write(' '.join(data)+'\n')

    # simple CP average
    def fusion_averager(self):
        self.F = (self.P + self.C) / 2

        








