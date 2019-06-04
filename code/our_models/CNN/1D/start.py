#!/usr/bin/env python3

print('This is an old file. Call crime_start.py instead. Exiting')
"""
import pickle
import sys
import os
import math
from time import sleep
import numpy as np

import Experiment

# import the model you want to use
import CNN 

# overall setting
data_dir = '/home/schin/Mercury/Chin/LA_crime/our_models/1D_CNN/'
# please NOTE only to make a level new dir, if you want use a multiple new dir you have to create the new dir
result_dir = '/home/schin/Mercury/Chin/Results/integrated_test/'

# parameter settings
settings = {}
settings['predict_day'] = 3 # predict 3 days ahead
settings['input_len'] = 30 
settings['normalize'] = False 
settings['pool_size'] = 2 
settings['model_type'] = 'CNN'
settings['model_subtype'] = 'CNN'
settings['label_as_signal'] = True 

settings['n_epoch'] = 3000
settings['l_r'] = 0.0000309 
settings['batch_size'] = 19 
#settings['dropout_rate'] = 0.5
settings['conv_layers'] = 3 
settings['num_filters'] = 50 
settings['first_pool'] = 1  # control pooling of CNN 
settings['second_pool'] = 1 
settings['l1_reg'] = 0
settings['drop_zeros'] = True 
settings['integrate_signals'] = True

# create comments for saving to results table 
settings['comment'] = ["integrated_test_for_real_day_2", 
                        "-", 
                        "-"
                        ]
# create an experiment
datasets = ['SY-MA.mat',
        'EG-MA.mat',
        'SY-NSA.mat',
        'EG-NSA.mat',
        'LB-MA.mat',
        'EG-CU.mat',
        'IQ-NSA.mat',
        'JO-CU.mat',
        'IQ-MA.mat',
        ]
# temp b/c I'm restarting
datasets = datasets[3:]

dropouts = np.random.rand(3 * len(datasets))
i = 0

for dataset in datasets:
    for run in range(3):
        for value in [True, False]:
            rate = dropouts[i]
            settings['dropout_rate'] = rate 
            settings['integrate_signals'] = value 
            settings['comment'][1] = str(rate) 
            settings['comment'][2] = str(value)
            print('Run: ' + str(run), end =  '    ')
            print('Dataset: ' + dataset, end='    ')
            print('Dropout: ' + str(rate), end='    ')
            print('Integrated? ' + str(value))
            #print('Days Forward: ' + str(days))

            experiment = Experiment.Experiment(data_dir=data_dir, set_filter=set_filter, 
                                                result_dir=result_dir+str(value)+'/'+str(run)+'/',
                                                model=CNN, 
                                                settings=settings,
                                                cases=[cases8[0]],
                                                restriction=dataset,
                                                save_to_table=True)
            experiment.run_experiment()
            experiment.record_save()

            print(experiment.MSE_ratios[0])
        i += 1





"""
