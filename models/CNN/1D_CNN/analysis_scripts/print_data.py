# load alll datasets and print them all out, just the data

import numpy as np
from scipy import io
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def load_and_print(data_dir, filename):
    # load
    labels = io.loadmat(data_dir + filename)['y']
    labels = np.reshape(labels, labels.shape[1])
    time_index = np.arange(0, labels.shape[0])

    print(labels.shape)
    print(time_index.shape)

    # print and save
    fig = plt.figure()
    plt.plot(time_index, labels)
    plt.savefig('/home/schin/Mercury/Chin/Results/spearmint/0731/display/'
            + filename[:-4] + '.png')
    plt.close(fig)

with open('/home/kxw/Desktop/Chin/Set_filter/long_list_0508.txt', 'rb') as f:
    datasets = pickle.load(f)

data_dir = '/home/schin/Mercury/Chin/DATA/2017-05-08_data/'

for dataset in datasets:
    load_and_print(data_dir, dataset)
