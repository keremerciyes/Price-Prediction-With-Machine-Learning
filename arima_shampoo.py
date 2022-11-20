from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

data = pd.read_csv("shampoo.csv")
#data.index = pd.date_range(start='1949-01', end='1960-11', freq='M')
data=data['Sales']
from statsmodels.tsa.stattools import adfuller

result = adfuller(data)

print('p-value: %f' % result[1])
print('p-value: %f' % result[2])


# # Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(data); ax1.set_title('Original Series'); ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(data.diff()); ax2.set_title('1st Order Differencing'); ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(data.diff().diff()); ax3.set_title('2nd Order Differencing')
#data2=data['1912':'1954']

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data.diff().dropna())

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data.diff().dropna()) 

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(data[3:], order=(2,1,1))
model_fit = model.fit()

predicted_data = model_fit.predict(start=4, end=len(data))
new_data = data[4:len(data)]

error = abs(np.divide((np.subtract(new_data, predicted_data)), new_data) * 100)
error_index = new_data.index


error_df = pd.DataFrame(error, error_index)

error_df.plot(title='error')

fig1, ax1 = plt.subplots()
ax = data.loc[1:].plot(ax=ax1)
predicted_data.plot(ax=ax)

residuals = pd.DataFrame(model_fit.resid)
fig2, ax2 = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax2[0])
residuals.plot(kind='kde',title='Density', ax=ax2[1])
plt.show()
