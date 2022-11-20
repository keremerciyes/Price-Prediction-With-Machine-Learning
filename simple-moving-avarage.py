# importing Libraries

# importing pandas as pd
from cmath import nan
from contextlib import closing
from optparse import Values
from time import time
import pandas as pd

# importing numpy as np
# for Mathematical calculations
import numpy as np


# Fixed size subset for simple moving avarage,  MA (time_period)
time_period = 10
moving_average = []
i = 0
nan_list = []


reliance = pd.read_csv('data-sets/BTC-USD-3M.csv', index_col='Date',
                       parse_dates=True)


closingPrice = reliance['Adj Close']
closingPrice = closingPrice.values

while i + time_period <= len(closingPrice):

    # Collecting closingPrice values in temp_prices list with time_period lenght of item.
    temp_prices = closingPrice[i: i + time_period]

    # Summing temp_prices list values and divide by time_period to find moving avarage.
    moving_average_value = sum(temp_prices) / time_period

    moving_average.append(moving_average_value)
    i += 1


print(len(moving_average))


# No data for first "time_period - 1" of element for calculation moving avarage, so for adding to 
for x in range(0, (time_period - 1), 1):
    nan_list.append(np.NaN)





moving_average = nan_list + moving_average

# print(len(nan_list))
# print(len(moving_average))


print(moving_average)
reliance = reliance.assign(MA =  moving_average)
print(reliance)
 

