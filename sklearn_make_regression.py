# coding: utf-8

from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=500, n_features=1, n_targets=1, noise=2)

plt.scatter(X, y)
plt.show()
