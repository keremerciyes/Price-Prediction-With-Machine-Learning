import numpy as np
from sklearn.decomposition import PCA

# create a time series dataset
time_series = np.array([[1,2,3,4,5], [2,4,6,8,10], [3,6,9,12,15]])

# initialize PCA with 2 principal components
pca = PCA(n_components=3)

# fit and transform the data
time_series_pca = pca.fit_transform(time_series.T)

# print the transformed data
print(pca.explained_variance_ratio_)