# code to print a lookup table for converting from blocks to latitudes and
#   longitudes

import numpy as np
import pandas as pd

min_lat = 33.6927
max_lat = 34.3837
min_long = -118.7051
max_long = -118.1157
lat_blocks = np.linspace(min_lat, max_lat, num=17)
long_blocks = np.linspace(min_long, max_long, num=17)

table = pd.DataFrame({'Block': range(256),
                        'Min_Latitude': np.zeros(256),
                        'Min_Longitude': np.zeros(256),
                        'Max_Latitude': np.zeros(256),
                        'Max_Longitude': np.zeros(256)})

print(table.head(10))

for block in range(256):
    lati = int(block // 16)
    longi = int(block % 16)

    table['Min_Latitude'][block] = lat_blocks[lati]  
    table['Min_Longitude'][block] = long_blocks[longi]  
    table['Max_Latitude'][block] = lat_blocks[lati+1]  
    table['Max_Longitude'][block] = long_blocks[longi+1]  

print(table.head(10))
table.to_csv('block_lookup_table.csv', index=False)
                        

