import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# create a time series dataset
time_series = np.array([[1,2,3,4,5], [2,4,6,8,10], [3,6,9,12,15]])

# initialize PCA with 2 principal components
pca = PCA()

# fit data
pca.fit_transform(time_series.T)

# print the transformed data
print(time_series.T)

plt.plot(np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()