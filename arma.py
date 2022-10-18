from cgi import print_form
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
register_matplotlib_converters()
from time import time

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

    
# read data
catfish_sales = pd.read_csv('data-sets/catfish.csv', parse_dates=[0], squeeze=True, index_col=0, date_parser=parser)

# data = pd.read_csv("data-sets/BTC-USD-3M.csv")

#infer the frequency of the data
catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))



start_date = datetime(2000,1,1)
end_date = datetime(2004,1,1)
lim_catfish_sales = catfish_sales[start_date:end_date]



for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(lim_catfish_sales.mean(), color='r', alpha=0.2, linestyle='--')


first_diff = np.diff(lim_catfish_sales)
print(len(first_diff))

plt.figure(figsize=(10,4))
plt.plot(first_diff)



# ACF & PACF
acf_vals = acf(first_diff)
pacf_vals = pacf(first_diff)


# Get training and testing sets
train_end = pd.to_datetime('2004-01-01')
test_end = pd.to_datetime('2005-01-01')


train_data = first_diff[5:10]
test_data = first_diff[20:30]


# Fit the ARMA Model
# define model
start  = time()
model = ARIMA(train_data, order=(4,0,1))
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)

#summary of the model
print(model_fit.summary())

#get prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

#get the predictions and residuals
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

plt.figure(figsize=(10,4))

plt.plot(test_data)
plt.plot(predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title('First Difference of BTC-USD', fontsize=20)
plt.ylabel('Price', fontsize=16)
plt.show()
