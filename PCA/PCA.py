from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# load the iris dataset
iris = load_iris()

# initialize PCA with two principal components
pca = PCA(n_components=1)

# fit and transform the data
iris_pca = pca.fit_transform(iris.data)

# print the explained variance ratio
print(pca.explained_variance_ratio_)