"""
Gavin Brown, Xiao Wang, Xiao Zhou

Checks for data tables in specificed directory.
If they exist, do nothing.
Otherwise, initialize them with list of columns. 
"""

import os

names = ['Model_Results']

columns ={ 
    names[0]: ['model_key', 'experiment_key', 'datetime',
        'comment1', 'comment2', 'comment3',
        'dataset', 'case_tuple', 'case',
        'compare_MSE', 'prediction_MSE', 'MSE_ratio',
        'model_type', 'model_subtype',
        'data_dir', 'result_dir',
        'predict_days', 'label_as_signal', 'input_len',
        'epochs', 'batch_size', 'learning_rate',
        'training_time', 'layers', 'num_params', 'extreme_detail'
        ]
}

path = './'

for name in names:
    filename = path + name + '.csv' 
    if os.path.isfile(filename):
        pass
    else:
        print('Creating table: ' + name)
        with open(filename, 'w') as f:
            for i in range(len(columns[name]) - 1):
                f.write(columns[name][i] + ' ')
            f.write(columns[name][-1]) # no trailing whitespace on last line
            f.write('\n')
            
