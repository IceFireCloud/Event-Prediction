"""
Script to compare two different datasets.
Right now, basic information.

Gavin Brown
"""

import numpy as np
from scipy import io
import sys
import os
import pickle
from time import sleep

base = '/home/schin/Mercury/Chin/DATA/'
dir1, dir2 = base+'2017-05-04_data/', base+'2017-07-25_data/'
dir3 = base+'2017-05-08_data/'
dir4 = base+'2017-09-18_data/'
dirs = [dir1, dir3, dir2, dir3]
names = ['0504', '0508', '0725', '0918']

with open('/home/kxw/Desktop/Chin/Set_filter/non_sparse.txt', 'rb') as f:
    datasets = pickle.load(f)

print(datasets)

for dataset in datasets:
    print(dataset)
    for i in range(4):
        d = dirs[i]
        data = io.loadmat(d + dataset + '.mat')
        X = data['X']
        y = data['y']

        # print the data date
        print(names[i], end=' ')

        # shapes
        print(X.shape, end='\t')

        # sparsity levels
        size = X.size
        non_zero = np.count_nonzero(X)
        print('%0.3f%% non-zero overall' % (100 * non_zero / size))
    print()




