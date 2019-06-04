"""
I think there may be more sparsity in the data than I expected.
This script will go through everything and find it!

Gavin Brown
"""

import numpy as np
import pickle
from scipy import io

data_dir = '/home/schin/Mercury/Chin/DATA/2017-05-08_data/'
with open('/home/kxw/Desktop/Chin/Set_filter/long_list_0508.txt', 'rb') as f:
    set_filter = pickle.load(f)

sparse = {
    'BH-MA',
    'BH-NSA',
    'EG-WCU',
    'JO-MA',
    'JO-NSA',
    'JO-WCU',
    'LB-NSA',
    'SA-MA',
    'SA-NSA'
}

"""
# overall treatment
print('Non-Sparse Datasets:')
for dataset in set_filter:
    if dataset[:-4] not in sparse:
        data = io.loadmat(data_dir + dataset)
        signals = data['X']
        count = 0
        total = signals.shape[1]
        for i in range(total):
            if np.max(signals[:, i]) != 0:
                count += 1
        print(dataset[:-4]+': '+str(count)+' of '+str(total)+' input signals are nonzero')

print('\nSparse Datasets:')
for dataset in set_filter:
    if dataset[:-4] in sparse:
        data = io.loadmat(data_dir + dataset)
        signals = data['X']
        count = 0
        total = signals.shape[1]
        for i in range(total):
            if np.max(signals[:, i]) != 0:
                count += 1
        print(dataset[:-4]+': '+str(count)+' of '+str(total)+' signals are nonzero')
"""    

# non-sparse cases where there are just a few non-sparse signals
few_signals = {
        'SY-NSA',
        'LB-MA',
        'JO-CU',
        'SY-MA'
        }

"""
print('Out of 490 total time steps, we have:\n')
for dataset in set_filter:
    if dataset[:-4] in few_signals:
        print(dataset[:-4]+' non-zero signals: ')
        data = io.loadmat(data_dir + dataset)
        signals = data['X']
        total = signals.shape[1]
        for i in range(total):
            if np.max(signals[:, i]) > 0:
                count = 0
                length = signals.shape[0]
                for j in range(length):
                    if signals[j, i] != 0: 
                        count += 1
                percent = 100 * count / length
                print('\t%i entries of signal %i are non-zero' % (count, i))

count = 0
total = 0
length = 490
solos = 0
duos = 0
others = 0

for dataset in set_filter:
    if dataset[:-4] not in sparse:
        data = io.loadmat(data_dir + dataset)
        signals = data['X']
        channels = signals.shape[1]
        for i in range(channels):
            if np.max(signals[:, i]) != 0:
                total += length
                entry_count = 0
                for j in range(length):
                    if signals[j, i] != 0: 
                        count += 1
                        entry_count += 1
                if entry_count == 1:
                    solos += 1
                elif entry_count == 2:
                    duos += 1
                else:
                    others += 1


print('Total: '+str(total))
print('Count: '+str(count))
print(490 * count / total)
print(str(solos)+' solos and '+str(duos)+' duos')
print(str(others)+' others')
"""






















