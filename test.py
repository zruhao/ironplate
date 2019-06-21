import glcm
import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing

testImgPath = './old/2.0.jpg'
modelPath = './models/model.m'

scale = False

img = cv2.imread(testImgPath)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

res = ()
for a, b in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
	mat = glcm.glcm(img_grey, a, b, gray_level=16)
	mat = glcm.unif(mat)
	res += glcm.calc(mat)
	print(res)

model = joblib.load(modelPath)

poly_reg = preprocessing.PolynomialFeatures(degree=4)
X_data = poly_reg.fit_transform(np.array([res]))
if scale:
    X_data = preprocessing.scale(X_data)

print((model.predict(X_data) + 1) / 2)
