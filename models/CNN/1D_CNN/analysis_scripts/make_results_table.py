"""
Script to read from a results directory and make a table, 
    with MSE ratios printed.
For now assume that there's just one set of results.

Gavin Brown
"""

#############################################################
# Change these as needed
#############################################################

result_dir = '/home/schin/Mercury/Chin/Results/spearmint/0904/redo_table/0/'
outfile = 'test_table.png'

#############################################################
# Don't need to adjust anything below
#############################################################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import io
import numpy as np

# grab directories for all cases
case_dirs = [result_dir + d + '/Normal/MAT/' for d in os.listdir(result_dir) if d != 'settings.txt']
case_dirs.sort()

datasets = os.listdir(case_dirs[0]) 
datasets.sort()

MSE_ratios = np.zeros((len(datasets), len(case_dirs)))

for i in range(len(case_dirs)):
    case_dir = case_dirs[i]
    for j in range(len(datasets)):
        dataset = datasets[j]
        data = io.loadmat(case_dir + dataset)

        T = data['truth']
        P = data['prediction']
        C = data['compare']

        C_MSE = np.mean((T - C)**2)
        P_MSE = np.mean((T - P)**2)
        MSE_ratio = P_MSE / C_MSE
        MSE_ratios[j, i] = '%0.2f' % MSE_ratio


# plot what we got
cols = [x[-13] for x in case_dirs]  # cols are case numbers
                                    # 13 b/c len('/Normal/MAT/') == 13

rows = [x[:-4] for x in datasets]

colors = []
for i in range(len(rows)):
    colors.append([])
    for j in range(len(cols)):
        if MSE_ratios[i, j] > 1:
            colors[i].append((1,0,0,0.5)) # red
        else: 
            colors[i].append((0.59,0.98,0.59,0.5)) # pale green

# actually plot
fig, ax = plt.subplots(figsize=(8, 2.5)) 

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.axis('off')

table = plt.table(cellText = MSE_ratios,
        cellColours=colors,
        rowLabels=rows,
        colLabels=cols,
        loc='center')

plt.savefig(outfile)
