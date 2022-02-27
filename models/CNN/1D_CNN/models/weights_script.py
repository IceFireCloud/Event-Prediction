"""
Script to explore the weight space of the saved Keras CNNs.
Currently working with "full model" of SY-MA, mega case

Gavin Brown
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import io

f = h5py.File('full_model.h5', 'r')

# access biases and weights of first convolutional layer
bias = np.array(f['model_weights']['conv1d_1']['conv1d_1']['bias:0'])
kernel = np.array(f['model_weights']['conv1d_1']['conv1d_1']['kernel:0'])

# kernel is (7, 24, 50) array
# that is, (filter_length, channels, num_filters)

# break out the data for show and tell
data = io.loadmat('/home/schin/Mercury/Chin/DATA/2017-05-08_data/SY-MA.mat')
signals, labels = np.transpose(data['X']), data['y']
signals = np.concatenate((signals, labels), axis=0)  # just like in main code
labels = labels.flatten()

start, end = 0, 488
x = np.arange(start, end)

# first, make a histogram of all the weights
if False:
    n, bins, patches = plt.hist(kernel.flatten(), bins=500)

    plt.xlabel('Weight')
    plt.ylabel('Count')
    plt.title('Distribution of First Layer Weights, Truncated')
    plt.xlim(-0.1, 0.1)
    plt.savefig('Overall_hist.png')

    plt.cla()

    # then, summary statistics of weights across input channels
    n, bins, patches = plt.hist(np.mean(kernel, axis=(0,2)), bins=30)
    plt.xlabel('Mean Weight')
    plt.ylabel('Count')
    plt.title('Means of Each Input Channel')
    #plt.xlim(-0.1, 0.1)
    plt.savefig('channel_hist.png')

    plt.cla()

    # fraction of weights within 0.025 of 0
    total = 7*24*50 
    inside = 0

    for num in kernel.flatten():
        if abs(num) <= 0.025:
            inside += 1

    print('Fraction of weights within 0.025 of 0')
    print(inside / total)

    # then, summary statistics of weights across days of filter
    bars = plt.bar(list(range(1,8)), np.mean(kernel, axis=(1,2)))
    plt.xlabel('Day of Filter')
    plt.ylabel('Mean Weight')
    plt.title('Means of Each Day, All Filters and Channels')
    #plt.xlim(-0.1, 0.1)
    plt.savefig('Filter_day_Bar.png')

    plt.cla()

    # want to check out these high and low input channels, indices 1 and 17
    #   (and channel 21, chosen at random for comparison)
    input_means = np.mean(kernel, axis=(0,2))
    for i in range(len(input_means)):
        print(str(i) + ': ' + str(input_means[i]))

    for i in [1, 17, 21]:
        n, bins, patches = plt.hist(kernel[:, i, :].flatten(), bins=30)
        plt.xlabel('Weight')
        plt.ylabel('Count')
        plt.title('Distribution of Weights for Input Number ' + str(i))
        #plt.xlim(-0.1, 0.1)
        plt.savefig('Input_%i_Weights_hist.png' % i)

        plt.cla()


    # Plot the two outlier channels - but nothing is there!
    plt.plot(x, labels[start:end], label='Truth')
    plt.plot(x, signals[1, start:end].flatten(), label='Channel 1')
    plt.plot(x, signals[17, start:end].flatten(), label='Channel 17')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Label Signal and Two Outlier Channels')
    plt.legend()
    plt.savefig('Label_Outliers_Time_Series.png')

    plt.cla()
    print(signals[17, start:end].flatten())

    # look at filter values for past seven days
    label_weights = kernel[:, 0, :]
    days = np.ones(50)
    for i in range(2, 8):
        days = np.concatenate((days, i * np.ones(50)))

    plt.plot(days, label_weights.flatten(), 'go', lw=0)
    plt.plot(np.arange(1, 8), np.mean(label_weights, axis=1), label='Mean')
    plt.xlabel('Day of Filter')
    plt.ylabel('Weight')
    plt.title('Weights on Label Input')
    plt.legend()
    plt.savefig('Input_Weights.png')
    plt.cla()

# Plot the signals as subplots
plt.figure(1)
for i in range(23):
    #plt.subplot(4,6,1+i)
    #plt.plot(x, signals[i, start:end].flatten())
    print(np.max(signals[i, start:end]))

#plt.savefig('All_Channels_Subplots.png')
#plt.cla()













