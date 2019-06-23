import os
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import joblib

dataPath = './data.csv'
modelPath = './models/model.m'
scalerPath = './models/scaler.m'

# 标准化
scale = True
trainSize = 0.75

data = np.genfromtxt(dataPath, delimiter=',')
np.random.shuffle(data)

x_data = data[:, 1: ]
# x_data = data[:, [5, 6, 8]]
y_data = [int(i * 2 - 0.5) for i in data[:, 0]]

poly_reg = preprocessing.PolynomialFeatures(degree=4)

X_data = poly_reg.fit_transform(x_data)
dl = int(data.shape[0] * trainSize)

if scale:
    scaler = preprocessing.StandardScaler()
    scaler_para = scaler.fit(X_data)
    X_data = scaler_para.transform(X_data)
    joblib.dump(scaler_para, scalerPath)

logistic = linear_model.LogisticRegression(penalty='l2')
logistic.fit(X_data[: dl], y_data[: dl])

joblib.dump(logistic, modelPath)

print(logistic.intercept_)
print(logistic.coef_)

print(logistic.score(X_data[: dl], y_data[: dl]))
print(logistic.score(X_data[dl:], y_data[dl:]))


# score = cross_val_score(logistic, X_data, y_data, cv=5)
# print(score)


# data2 = np.genfromtxt('./data2.csv', delimiter=',')
# x_test = data2[:, 1:]
# X_test = poly_reg.fit_transform(x_test)
# y_test = [int(i * 2 - 0.5) for i in data2[:, 0]]
# print(logistic.score(X_test, y_test))

# res = logistic.predict(X_data[: dl])
# print(res)
# print(y_data[: dl])
