# util functions for vision transformer
# Created: 6/16/2021
# Status: in progress

import json
import csv

import numpy as np

def read_json(filename):
    with open(filename) as buf:
        return json.loads(buf.read())

def read_csv_cox(filename):
    with open(filename, 'r') as f:
        # reader = csv.reader(f)
        # your_list = list(reader)
        reader = csv.DictReader(f)
        fileIDs, time_obs, time_hit = [], [], []
        for r in reader:
            temp = r['PROGRESSES']
            # temp = r['PROGRESSION_CATEGORY']
            # temp = r['TIME_TO_PROGRESSION']
            if len(temp) == 0:
                continue
            else:
                fileIDs += [str(int(float(r['RID'])))]
                # time_obs += [int(float(r['TIME_TO_FINAL_DX']))]
                time_obs += [int(float(r['TIMES_ROUNDED']))]
                try:
                    time_hit += [int(float(r['PROGRESSES']))]
                except:
                    time_hit += [0]
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    return fileIDs, time_obs, time_hit

def rescale(array, tup):
    m = np.min(array)
    if m < 0:
        array += -m
    a = np.max(array)-np.min(array)
    t = tup[1] - tup[0]
    return array * t / a
