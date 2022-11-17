from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm

# path = 'data-sets/air_passengers.csv'
# data = pd.read_csv(path, index_col='Month')

data = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
data.index = pd.date_range(start='1700', end='2009', freq='A')
data.head(20)


from statsmodels.tsa.stattools import adfuller
result = adfuller(data['SUNACTIVITY'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))


# # Original Series
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# ax1.plot(data.Passengers); ax1.set_title('Original Series'); ax1.axes.xaxis.set_visible(False)
# # 1st Differencing
# ax2.plot(data.Passengers.diff()); ax2.set_title('1st Order Differencing'); ax2.axes.xaxis.set_visible(False)
# # 2nd Differencing
# ax3.plot(data.Passengers.diff().diff()); ax3.set_title('2nd Order Differencing')




from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(data.Passengers.diff().dropna())

from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(data.Passengers.diff().dropna())


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data, order = (2,1,2))
model_fit = model.fit()


from statsmodels.graphics.tsaplots import plot_predict

fig, ax = plt.subplots()
ax = data.loc['1900':].plot(ax=ax)

# I did not use this function yet.
# predicted_data = model_fit.get_prediction(start="1950", end="2009")

plot_predict(model_fit, start="1950", end="2009", dynamic=False, ax=ax, plot_insample=True)
plt.show()





