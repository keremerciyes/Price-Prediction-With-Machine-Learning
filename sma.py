import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Initialize an empty list to store moving averages
moving_averages = []

data=pd.read_csv("XOM.csv")

closingPrice = data['Adj Close']
closingPrice=closingPrice.values


# every window of size ... (sma's rank)
window_size = 200

# Loop through the array to consider
i = 0

for i in range(len(closingPrice)-window_size+1):

    # Store elements from i to i+window_size
    # in list to get the current window
    window = closingPrice[i : i + window_size]

    
    # Calculate the average of current window
    window_average = sum(window) / window_size

    
    # Store the average of current
    moving_averages.append(window_average)  

#print(moving_averages)

nan_list = []
for x in range(0, window_size-1,1):
    nan_list.append(np.NaN)

err = []
for x in range(0,len(moving_averages)) :
    err_val = abs(moving_averages[x]-closingPrice[x+window_size-1])/closingPrice[x+window_size-1]*100
    err.append(err_val)

# adding nan list to the start of the array 
err = nan_list + err

# adding nan list to the start of the array 
moving_averages = nan_list + moving_averages

data=data.assign(Moving_average=pd.Series(moving_averages,index=data.index))
data=data.assign(Error=pd.Series(err,index=data.index))
fig1=plt.figure(figsize=(10,8))
fig2=plt.figure(figsize=(10,8))
ax1=fig1.add_subplot(111)
ax2=fig2.add_subplot(111)
data['Adj Close'].plot(ax=ax1, color='b', lw=3, legend=True)
data['Moving_average'].plot(ax=ax1, color='g', lw=3, legend=True)
data['Error'].plot(ax=ax2, color='r', lw=3, legend=True)
plt.show()
