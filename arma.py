import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
register_matplotlib_converters()
from statsmodels.graphics.tsaplots import plot_acf


data = pd.read_csv("data-sets/BTC-USD-3M.csv")


closingPrice = data['Adj Close']
closingPrice = closingPrice.values

plt.figure(figsize=(10,4))
plt.plot(closingPrice)


plt.axhline(closingPrice.mean(), color='r', alpha=0.2, linestyle='--')

first_diff = np.diff(closingPrice)

print(first_diff)

plt.figure(figsize=(10,4))
plt.plot(first_diff)

acf_vals = acf(first_diff, 100)
print(acf_vals)
plt.show()
