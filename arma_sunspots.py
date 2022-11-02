import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.api import qqplot

print(sm.datasets.sunspots.NOTE)

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
dta.index.freq = dta.index.inferred_freq
del dta["YEAR"]

dta.plot(figsize=(12, 8))


fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

arma_mod20 = ARIMA(dta, order=(3, 0, 0)).fit()
print(arma_mod20.params)
sm.stats.durbin_watson(arma_mod20.resid.values)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax = arma_mod20.resid.plot(ax=ax)


resid = arma_mod20.resid

stats.normaltest(resid)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line="q", ax=ax, fit=True)


fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

r, q, p = sm.tsa.acf(resid.values.squeeze(), fft=True, qstat=True)
data = np.c_[np.arange(1, 25), r[1:], q, p]

table = pd.DataFrame(data, columns=["lag", "AC", "Q", "Prob(>Q)"])
print(table.set_index("lag"))


predict_sunspots = arma_mod20.predict("1920", "1970", dynamic=True)
print(predict_sunspots)
fig = plt.figure(figsize=(12, 8))
ax3 = fig.add_subplot(111)
plt.plot(predict_sunspots)

def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

plt.show()