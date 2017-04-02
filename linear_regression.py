# coding: utf-8

from sklearn import datasets
from sklearn.linear_model import LinearRegression

data = datasets.load_boston()

data_X = data.data
data_y = data.target

print data_X.shape

lr = LinearRegression()
lr.fit(data_X, data_y)

print lr.predict(data_X[: 5])
print data_y[: 5]
