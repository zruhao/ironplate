import os
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.externals import joblib

dataPath = './data.csv'
modelPath = './models/model.m'

# 标准化
scale = False
trainSize = 0.75

data = np.genfromtxt(dataPath, delimiter=',')
np.random.shuffle(data)

# x_data = data[:, 1:]
x_data = data[:, [5, 6, 8]]
y_data = [int(i * 2 - 0.5) for i in data[:, 0]]

poly_reg = preprocessing.PolynomialFeatures(degree=7)

X_data = poly_reg.fit_transform(x_data)
dl = int(data.shape[0] * trainSize)

if scale:
    X_data = preprocessing.scale(X_data)

logistic = linear_model.LogisticRegression()
logistic.fit(X_data[: dl], y_data[: dl])

joblib.dump(logistic, modelPath)

print(logistic.intercept_)
print(logistic.coef_)

print(logistic.score(X_data[: dl], y_data[: dl]))
print(logistic.score(X_data[dl:], y_data[dl:]))
