from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

data = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
data.index = pd.date_range(start='1700', end='2009', freq='A')

from statsmodels.tsa.stattools import adfuller
result = adfuller(data['SUNACTIVITY'])
# print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
#   print('\t%s: %.3f' % (key, value))

# from statsmodels.tsa.seasonal import seasonal_decompose

# result = seasonal_decompose(data, model='additive')
# result.plot()
# plt.show()

# # Original Series
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# ax1.plot(data.SUNACTIVITY); ax1.set_title('Original Series'); ax1.axes.xaxis.set_visible(False)
# # 1st Differencing
# ax2.plot(data.SUNACTIVITY.diff()); ax2.set_title('1st Order Differencing'); ax2.axes.xaxis.set_visible(False)
# # 2nd Differencing
# ax3.plot(data.SUNACTIVITY.diff().diff()); ax3.set_title('2nd Order Differencing')
data2=data['1912':'1954']
data2=data2['SUNACTIVITY']
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data2.dropna())

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data2.dropna()) 

# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(data.SUNACTIVITY.dropna())

# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(data.SUNACTIVITY.dropna())

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(data['1900':], seasonal_order = (3,0,2,11))
model_fit = model.fit()

predicted_data = model_fit.predict(start="1950", end="2007")
new_data = data['1950':'2007']
new_data = new_data['SUNACTIVITY']

error = abs(np.divide((np.subtract(new_data.values, predicted_data.values)), new_data.values) * 100)
error_index = new_data.index


error_df = pd.DataFrame(error, error_index)

error_df.plot(title='error')

fig1, ax1 = plt.subplots()
ax = data.loc['1900':].plot(ax=ax1)
predicted_data.plot(ax=ax)

residuals = pd.DataFrame(model_fit.resid)
fig2, ax2 = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax2[0])
residuals.plot(kind='kde',title='Density', ax=ax2[1])
plt.show()

# plot_predict(model_fit, start="1950", end="2009", dynamic=False, ax=ax, plot_insample=True)