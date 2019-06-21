import os
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.externals import joblib

dataPath = './data.csv'
modelPath = './models/model.m'

# 标准化
scale = False

data = np.genfromtxt(dataPath, delimiter=',')
np.random.shuffle(data)

x_data = data[:, 1:]
y_data = [i * 2 - 1 for i in data[:, 0]]

poly_reg = preprocessing.PolynomialFeatures(degree=4)

X_data = poly_reg.fit_transform(x_data)

if scale:
    X_data = preprocessing.scale(X_data)

logistic = linear_model.LogisticRegression()
logistic.fit(X_data[: 150], y_data[: 150])

joblib.dump(logistic, modelPath)

print(logistic.intercept_)
print(logistic.coef_)

print(logistic.score(X_data[: 150], y_data[: 150]))
print(logistic.score(X_data[150:], y_data[150:]))
