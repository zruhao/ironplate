import glcm
import cv2
import numpy as np
import joblib
from sklearn import preprocessing

testImgPath = './new/2.5.jpg'
modelPath = './models/model.m'
scalerPath = './models/scaler.m'

scale = True

img = cv2.imread(testImgPath)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

res = ()
for a, b in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
	mat = glcm.glcm(img_grey, a, b, gray_level=16)
	mat = glcm.unif(mat)
	res += glcm.calc(mat)
	print(np.array([res]))

model = joblib.load(modelPath)

poly_reg = preprocessing.PolynomialFeatures(degree=4)
X_data = poly_reg.fit_transform(np.array([res]))
if scale:
    scaler_para = joblib.load(scalerPath)
    X_data = scaler_para.transform(X_data)

print(int(model.predict(X_data) + 1.5) / 2)
