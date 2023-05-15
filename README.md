# Price Prediction with Machine Learning

This repository contains machine learning models for predicting prices using various machine learning techniques.

## Folder Structure

- **Linear Regression**: This directory contains Jupyter notebooks implementing linear regression models. 
- **Support Vector Regression**: This directory contains Jupyter notebooks implementing support vector regression models.
- **Decision Tree Regression**: This directory contains Jupyter notebooks implementing decision tree regression models.
- **Random Forest Regression**: This directory contains Jupyter notebooks implementing random forest regression models.

Each directory contains multiple notebooks that apply the respective machine learning model to different datasets or configurations.

## Getting Started

To run the notebooks, you'll need Jupyter and the following Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`. Install them using pip:

```pip install jupyter pandas numpy matplotlib seaborn sklearn```

Then, clone this repository, navigate to the desired directory, and launch Jupyter:

```git clone https://github.com/oktaykurt/price-prediction-with-machine-learning.git```

```cd price-prediction-with-machine-learning/Linear\ Regression```

```jupyter notebook```


Open the desired notebook in your browser.

## Usage

Each notebook contains code cells that you can run one by one. They follow this general structure:

1. **Import libraries**: Import the necessary Python libraries.
2. **Load the dataset**: Load a CSV or Excel file into a pandas DataFrame.
3. **Preprocess the data**: Clean the data and prepare it for the machine learning model.
4. **Train the model**: Use a scikit-learn model (like `LinearRegression` or `SVR`) to train the model on the data.
5. **Evaluate the model**: Evaluate the model's performance using metrics like mean squared error.

Remember to replace the dataset path with the correct path to the dataset on your machine.

