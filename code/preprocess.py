# code to replicate the preprocessing done in Bertozzi et al.
# This includes:
#   Integrating crimes by day
#   Applying cubic spline superresolution, whatever that actually means
#   Add extra information such as day of week, weather, or holiday flag

import pandas as pd
import datetime

df = pd.read_csv('../data/Event_Counts.csv')

df.Date = pd.to_datetime(df.Date)

# Integrate crimes by day, a running sum that's reset at midnight

# Cubic spline superresolution?

# Add additional information
df.Week = df.Date.dt.weekday_name

# Write out
df.to_csv('../data/Bertozzi_Processed.csv', index=False)
