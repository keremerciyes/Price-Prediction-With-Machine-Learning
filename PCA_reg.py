from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load the Boston Housing dataset
boston = fetch_california_housing()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)

# initialize PCA with 3 principal components
pca = PCA(n_components=3)

# fit and transform the training data
X_train_pca = pca.fit_transform(X_train)

# initialize the linear regression model
model = LinearRegression()

# fit the model on the transformed training data
model.fit(X_train_pca, y_train)

# transform the testing data
X_test_pca = pca.transform(X_test)

# make predictions on the transformed testing data
y_pred = model.predict(X_test_pca)

# calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")