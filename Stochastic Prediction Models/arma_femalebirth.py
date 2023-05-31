from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

data = pd.read_csv("femalebirths.csv")

data=data['Births']
from statsmodels.tsa.stattools import adfuller

result = adfuller(data)

print('p-value: %f' % result[1])

# # Original Series
fig, ax1 = plt.subplots(1)
ax1.plot(data); ax1.set_title('Original Series'); 


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data.dropna())

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data.dropna()) 

# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(data.SUNACTIVITY.dropna())

# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(data.SUNACTIVITY.dropna()) 

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(data[3:], order = (1,0,1))
model_fit = model.fit()

predicted_data = model_fit.predict(start=7, end=len(data))
new_data = data[7:len(data)]

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