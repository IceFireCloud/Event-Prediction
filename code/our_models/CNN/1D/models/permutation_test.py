"""
Test the importance of various inputs by permuting them one at a time.
We have saved models and can load the test data.
Here we use SY-MA, mega case.
Permute two inputs at a time and track changes in the MSE.

With 24 inputs, there are 24*23/2=276 possible combinations.
Save the results in a CSV.

Gavin Brown
"""

import numpy as np
from scipy import io
import pandas as pd

from keras.models import load_model

# create set of permutations, including 
combos = set() 
for i in range(24):
    for j in range(24):
        if i != j and frozenset([i, j]) not in combos:
            combos.add(frozenset([i, j]))

# load model
model = load_model('no_label_model.h5')

# load and configure testing data
data = io.loadmat('/home/schin/Mercury/Chin/DATA/2017-05-08_data/SY-MA.mat')
signals, labels = np.transpose(data['X']), data['y']
#signals = np.concatenate((signals, labels), axis=0)  # just like in main code
labels = labels.flatten()
print(signals.shape)

def create_data(signals, labels, permute=[0, 0], drop=None):
    copy = signals.copy()
    signals[permute[0], :] = copy[permute[1], :] 
    signals[permute[1], :] = copy[permute[0], :] 

    if drop:
        signals[drop, :] = np.zeros(signals[drop, :].shape)

    test_x = []
    test_y = []
    # copied from ../base_model.py
    label_pointer = 357
    pointer = label_pointer - 30
    while label_pointer <= 489:
        test_x_piece = signals[:, pointer:pointer + 30]
        test_x_piece = np.transpose(test_x_piece)
        test_x.append(test_x_piece)

        test_y_piece = labels[label_pointer]
        test_y.append(test_y_piece)

        label_pointer += 1
        pointer += 1
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return [test_x, test_y]
"""
# run on first test
[test_x, test_y] = create_data(signals, labels)
prediction = model.predict(test_x, batch_size=test_x.shape[0])
prediction = np.sum(prediction, axis=1)
base_MSE = np.sum((prediction - labels[357:])**2)/len(prediction)

# create the CSV
f = open('no_label_model_dropping_results.csv', 'w')
f.write('Channel,MSE_ratio\n')

# run on all permutations
for combo in combos:
    permute = list(combo)
    [test_x, test_y] = create_data(signals, labels, permute)
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    prediction = np.sum(prediction, axis=1)
    MSE = np.sum((prediction - labels[357:])**2)/len(prediction)

    # add to CSV
    f.write(str(permute[0])+','+str(permute[1])+','+str(MSE / base_MSE)+'\n')

# run on all channels
for channel in range(23):
    [test_x, test_y] = create_data(signals, labels, drop=channel)
    prediction = model.predict(test_x, batch_size=test_x.shape[0])
    prediction = np.sum(prediction, axis=1)
    MSE = np.sum((prediction - labels[357:])**2)/len(prediction)

    f.write(str(channel)+','+str(MSE / base_MSE)+'\n')

f.close
"""

















