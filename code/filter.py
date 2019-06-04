# Code to do initial processing and filtering of LA crime data
# Restrict time, remove columns, etc.

import numpy as np
import pandas as pd
import datetime

# read in data
data = pd.read_csv('./data/Raw_2010_to_2017.csv')

# toss unneeded columns
data = data[['Date Occurred',
            'Time Occurred',
            'Location ']]
data.columns = ['Date', 'Time', 'Location']
print('Loaded', data.shape[0], 'rows.')

# restrict dates
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
start_date, end_date = datetime.date(2015, 7, 1), datetime.date(2016, 1, 1)
mask = (data['Date'] >= start_date) & (data['Date'] < end_date)
data = data.loc[mask]
print('Date restricted to', data.shape[0], 'rows.')

# adjust location formats
## strip parentheses
data['Location'] = data['Location'].str.slice(1, -1)

## split into two columns, concatenate back
data = data.join(data['Location'].str.split(',', expand=True))
data = data[['Date', 'Time', 0, 1]]
data.columns = ['Date', 'Time', 'Lat', 'Long']

## adjust formats
data.Lat = pd.to_numeric(data.Lat)
data.Long = pd.to_numeric(data.Long)

# restrict location, as per paper
min_lat = 33.6927
max_lat = 34.3837
min_long = -118.7051
max_long = -118.1157
mask = ((data.Lat>=min_lat)&
        (data.Lat<=max_lat)&
        (data.Long>=min_long)&
        (data.Long<=max_long))
data = data.loc[mask]

"""
print('Read in', rolled_up.shape[0], 'rows and', 
np.sum(rolled_up.Count), 'events.')
print('Highest event count is', np.max(rolled_up.Count))

# remove extraneous columns, only need these few
del rolled_up['Lat_Block']
del rolled_up['Long_Block']

# reshape so the block numbers are their own column
unstacked = rolled_up.pivot_table(index=['Date', 'Hour', 'Training_Split'],
                                columns='Overall_Block',
                                values='Count',
                                fill_value=0)   # fill NaN as 0, no crime

# check that we preserve the number of events
count = 0
hi = 0

for i in range(256):
if i in unstacked.columns:
    count += np.sum(unstacked[i])
    if hi < np.max(unstacked[i]):
        hi = np.max(unstacked[i])

print('Total:', count)
print('Max:', hi)

# this is what we need to start; save it out
unstacked.to_csv('./data/Block_Columns.csv')

rolled_up.to_csv('./data/Event_Counts.csv', index=False)
"""







































