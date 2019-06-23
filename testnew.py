import glcm
import cv2
import numpy as np
import joblib
from sklearn import preprocessing

csvPath = './data1.csv'
modelPath = './models/model.m'
scalerPath = './models/scaler.m'

scale = True

res = np.genfromtxt(csvPath, delimiter=',')
x_data = res[:, 1:]
y_data = res[:, 0]

model = joblib.load(modelPath)

poly_reg = preprocessing.PolynomialFeatures(degree=4)
X_data = poly_reg.fit_transform(np.array(x_data))
if scale:
    scaler_para = joblib.load(scalerPath)
    X_data = scaler_para.transform(X_data)

pre = model.predict(X_data)

ans = [int(i * 2 - 0.5) == j for i, j in zip(y_data, pre)]
print(sum(ans) / len(ans))

# print(pre)
# print(y_data)
