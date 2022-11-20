import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

rcParams['figure.figsize'] = 18, 5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.prop_cycle'] = cycler(color=['#365977'])
rcParams['lines.linewidth'] = 2.5


# # Declare
# white_noise = np.random.randn(1000)

# # Plot
# plt.title('White Noise Plot', size=20)
# plt.plot(np.arange(len(white_noise)), white_noise)


# Start with a random number - let's say 0
random_walk = [0]

for i in range(1, 1000):
    # Movement direction based on a random number
    num = -1 if np.random.random() < 0.5 else 1
    random_walk.append(random_walk[-1] + num)
    
    
# Plot
plt.title('Random Walk Plot', size=20)
plt.plot(np.arange(len(random_walk)), random_walk)
plt.show()
