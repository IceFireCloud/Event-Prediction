# Gavin Brown, grbrown@bu.edu
# Code to reshape data and add columns, prep for use in learning

import numpy as np
import pandas as pd
import datetime

df = pd.read_csv('./data/Event_Counts.csv')
print('Read in', df.shape[0], 'rows and', 
    np.sum(df.Count), 'events.')
print('Highest event count is', np.max(df.Count))

# remove extraneous columns, only need these few
del df['Lat_Block']
del df['Long_Block']

# reshape so the block numbers are their own column
df = df.pivot_table(index=['Date', 'Hour', 'Training_Split'],
                                    columns='Overall_Block',
                                    values='Count',
                                    fill_value=0)   # fill NaN as 0, no crime

df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')

df.Weekday = df.Date.dt.weekday


