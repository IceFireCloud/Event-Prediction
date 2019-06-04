# code to select the days where everything bad happened

import pandas as pd
import numpy as np
import datetime

# read in data
data = pd.read_csv('./data/Raw_2010_to_2017.csv')

# restrict dates
data['Date Occurred'] = pd.to_datetime(data['Date Occurred'], format='%m/%d/%Y')
start_date = datetime.date(2015, 10, 15)
mask = data['Date Occurred'] == start_date
data = data.loc[mask]

# adjust location formats
## strip parentheses
data['Location '] = data['Location '].str.slice(1, -1)

## split into two columns, concatenate back
data = data.join(data['Location '].str.split(',', expand=True))

## adjust formats
data[0] = pd.to_numeric(data[0])
data[1] = pd.to_numeric(data[1])

# restrict location
min_lat = 34.21095
max_lat = 34.25414
min_long = -118.631
max_long = -118.59
mask = ((data[0]>=min_lat)&
        (data[0]<max_lat)&
        (data[1]>=min_long)&
        (data[1]<max_long))
data = data.loc[mask]

data.to_csv('./data/bad_days.csv', index=False)

