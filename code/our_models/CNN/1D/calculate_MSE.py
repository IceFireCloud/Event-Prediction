# Event Prediction
# Gavin Brown, Xiao Wang, Xiao Zhou

# Code to take in a result directory and output mean and median MSE ratios comparing prediction
#   and true label

import os
from scipy import io
import numpy as np

def calculate_one_file(filename):
    """return MSE ratio(s), given filename with path"""
    data = io.loadmat(filename)
    T, C = data['truth'][0], data['compare'][0]
    n = len(T)

    compare_MSE = sum((T - C)**2) / n
    if compare_MSE == 0:
        return(None)

    # track MSEs for prediction, fusion, etc
    other_MSEs = {}
    others = [x for x in list(data.keys()) if x[0]!='_' and x not in {'truth', 'time_index', 'compare'}]
    for other in others:
        other_MSEs[other] = sum((T - data[other][0])**2) / n

    # track ratios for prediction, fusion, etc
    MSE_ratios = {}
    for key, value in other_MSEs.items():
        MSE_ratios[key] = value / compare_MSE

    return(MSE_ratios)

def find_MSE_ratios(path):
    """given directory, calculate mean and medians"""
    MSE_ratios = []
    queue = [x for x in os.walk(top=path)]  # see os.walk documentation, it generates file tree          
    for item in queue:
        path, filenames = item[0], item[2]  # path and filenames sitting in that path
        mat_files = [path+'/'+f for f in filenames if f[-4:]=='.mat']
        for mat_file in mat_files:
            MSE_ratios.append(calculate_one_file(mat_file))

    # need to remove the None's
    MSE_ratios = [x for x in MSE_ratios if x is not None]

    # transform and calculate means, medians
    ratio_lists = {}
    statistics = {} 
    for key in MSE_ratios[0].keys():
        ratio_lists[key] = []
        statistics[key] = {}
        for i in range(len(MSE_ratios)):
            ratio_lists[key].append(MSE_ratios[i][key])
        statistics[key]['mean'] = np.mean(ratio_lists[key]) 
        statistics[key]['median'] = np.median(ratio_lists[key]) 
        
    return(statistics)

output = find_MSE_ratios('/home/schin/Mercury/Chin/Results/0619_nnFusion/case5')

for key, value in output.items():
    print(key, '\n', value)

