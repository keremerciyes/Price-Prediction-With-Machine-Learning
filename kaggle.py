from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# sns.set()
# plt.style.use('ggplot')
# plt.rcParams['figure.figsize'] = (14, 8)

# # Draw samples from a standard Normal distribution (mean=0, stdev=1).
# points = np.random.standard_normal(1000)

# # making starting point as 0
# points[0]=0

# # Return the cumulative sum of the elements along a given axis.
# random_walk = np.cumsum(points)
# random_walk_series = pd.Series(random_walk)4

random_walk=pd.read_csv("XOM.csv")
closingPrice = random_walk['Adj Close']
closingPrice=closingPrice.values
print(len(closingPrice))


fig3=plt.figure(figsize=(10,8))
ax3=fig3.add_subplot(111)
random_walk['Adj Close'].plot(ax=ax3, color='b', lw=3, legend=True)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure


from statsmodels.tsa.arima_process import ArmaProcess

# start by specifying the lag
ar3 = np.array([3])

# specify the weights : [1, 0.9, 0.3, -0.2]
ma3 = np.array([1, 0.9, 0.3, -0.2])

# simulate the process and generate 1000 data points
MA_3_process = ArmaProcess(ar3, ma3).generate_sample(nsample=1258)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure
plt.plot(MA_3_process)
plt.title('Simulation of MA(3) Model')
plt.show()
plot_acf(MA_3_process, lags=20);

ar3 = np.array([1, 0.9, 0.3, -0.2])
ma = np.array([3])
simulated_ar3_points = ArmaProcess(ar3, ma).generate_sample(nsample=1258)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure
plt.plot(simulated_ar3_points)
plt.title("Simulation of AR(3) Process")
plt.show()
plot_acf(simulated_ar3_points);

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(simulated_ar3_points);

from statsmodels.tsa.stattools import pacf

pacf_coef_AR3 = pacf(simulated_ar3_points)
print(pacf_coef_AR3)

# ARMA(1,1)
ar1 = np.array([1, 0.6])
ma1 = np.array([1, -0.2])
simulated_ARMA_1_1_points = ArmaProcess(ar1, ma1).generate_sample(nsample=1258)
plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(simulated_ARMA_1_1_points)
plt.title("Simulated ARMA(1,1) Process")
plt.xlim([0, 200])
plt.show()

plot_acf(simulated_ARMA_1_1_points);
plot_pacf(simulated_ARMA_1_1_points);

# ARMA(2,2)
ar2 = np.array([1, 0.6, 0.4])
ma2 = np.array([1, -0.2, -0.5])

simulated_ARMA_2_2_points = ArmaProcess(ar2, ma2).generate_sample(nsample=1258)
plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(simulated_ARMA_2_2_points)
plt.title("Simulated ARMA(2,2) Process")
plt.xlim([0, 200])
plt.show()

plot_acf(simulated_ARMA_2_2_points);
plot_pacf(simulated_ARMA_2_2_points);
