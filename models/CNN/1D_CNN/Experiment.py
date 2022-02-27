import pickle
import os
import sys
from math import isnan
from statistics import median
import numpy as np
import json
from datetime import datetime

# we call running NN for all cases and for all datasets with fixed hyper-parameter setting an experiment
# the attributes of an experiments includes : data_dir, set_filter, result_dir, model, settings
# data_dir : the directory storing the data(all data sets)
# set_filter: the sets in the data dir that we want to run experiments on, it is a list, e.g. long_list.txt
# result_dir : the directory storing the results. the results should have 3 forms, .mat, .csv, .png
# Model
# cases: training/testing splitting cases
# settings: settings of the experiment, a dictionary, inlcluding :
# predict day: the day you want to predict. 0=today, 1 = tomorrow, etc
# input_len : how many days of history that our prediction is based on(the x axis of the input image of the CNN)
# n_epoch : number of epoch
# l_r : learning rate
# batch_size: mini-batch

class Experiment(object):
    def __init__(self, data_dir, set_filter, result_dir, model, cases, settings,
            restriction=None, save_to_table=False):
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.model = model
        self.settings = settings
        self.cases = cases
        self.MSE_ratios = []   
        self.total_train_time = 0
        self.mean_MSE_ratio = None
        self.median_MSE_ratio = None
        self.index = None 
        self.save_to_table = save_to_table
        if restriction:
            self.set_filter = [x for x in set_filter if x==restriction]
        else:
            self.set_filter = set_filter

    def get_new_index(self, key='Model'):
        """return the index of the last experiment, +1, hope we don't have to runs try at the same time"""
        data = np.loadtxt('Model_Results.csv', dtype='str') 
        if data.ndim == 1:
            index = 0  
            ncol = data.shape[0]
        else: 
            ncol = data.shape[1]
            with open('Model_Results.csv', 'r') as f:
                for line in reversed(f.readlines()):
                    entries = line.split()
                    m_key, e_key = entries[0], entries[1]
                    if key == 'Experiment':
                        index = int(e_key) + 1
                        break
                    elif key == 'Model' and m_key != 'dummy':
                        if m_key == 'model_key':
                            index = 0
                        else:
                            index = int(m_key) + 1
                        break

        # if we're starting an experiment, write a dummy line
        if key == 'Experiment':
            with open('Model_Results.csv', 'a') as f:
                f.write('dummy ')
                f.write(str(index))
                f.write((ncol-2)*' dummy' + '\n') # need space for correct number of columns

        return(index)

    def delete_dummy(self):
        """delete the line of dummy data corresponding to our experiment"""
        with open('Model_Results.csv', 'r') as f:
            lines = f.readlines() 

        for i in range(len(lines)):
            words = lines[i].split()
            if words[1] == str(self.experiment_index):
                if words[0] == 'dummy':
                    lines = lines[:i] + lines[i+1:]
                    break
        
        # rewrite whole file.
        with open('Model_Results.csv', 'w') as f:
            for line in lines:
                f.write(line)
         
    def run_experiment(self):
        if self.save_to_table:
            self.experiment_index = self.get_new_index(key='Experiment')

        # loop: every cases
        for case in self.cases:
            # for every dataset
            for file_name in self.set_filter:
                # create an object of the model
                model = self.model.Model(data_dir=self.data_dir, file_name=file_name,
                                         result_dir=self.result_dir, case=case, 
                                         settings=self.settings)
                # do data processing
                model.data_processing()
                # run the model
                self.keras_model = model.run_model()
                # compute fusion signal
                model.fusion_averager()
                # save result for this model
                model.output_result({'compare': model.C, 'prediction': model.P}, 'Normal')
                if self.save_to_table:
                    model_index = self.get_new_index(key='Model')
                    model.save_settings(model_index, self.experiment_index)
                # add to total training time
                self.total_train_time += model.train_time
                # add to MSE ratios
                if not isnan(model.MSE_ratio):
                    self.MSE_ratios.append(model.MSE_ratio)

        self.mean_MSE_ratio = sum(self.MSE_ratios) / len(self.MSE_ratios)
        self.median_MSE_ratio = median(self.MSE_ratios)

        if self.save_to_table:
            self.delete_dummy() # delete placeholder from data table 

    def record_save(self):
        # record settings to settings.txt
        try:
            sys.stdout = open(self.result_dir + 'settings' + '.txt', 'w+')
        except:
            os.mkdir(self.result_dir)
            sys.stdout = open(self.result_dir + 'settings' + '.txt', 'w+')

        print('prediction day:' + str(self.settings['predict_day']))
        print('input_length:' + str(self.settings['input_len']))
        print('number of epoch:' + str(self.settings['n_epoch']))
        print('learning rate:' + str(self.settings['l_r']))
        print('batch_size:' + str(self.settings['batch_size']))
        print('mean_MSE_ratio:' + str(self.mean_MSE_ratio))
        print('median_MSE_ratio:' + str(self.median_MSE_ratio))
        print('total_train_time:' + str(self.total_train_time))
        print('layers:' + str(self.settings['conv_layers']))
        print('dropout rate:' + str(self.settings['dropout_rate']))
        print('Max Pooling:' + str(self.settings['pool_size']))

        # record information to Experiment_Runs.py
        sys.stdout = sys.__stdout__

