import cv2
import numpy as np
import os
import getfileslist as gfl

oldPath = './data/'
newPath = './new/'

# 统一路径名格式
if oldPath[-1] != '/':
	oldPath += '/'
if newPath[-1] != '/':
	newPath += '/'

filesList = gfl.getFilesList(oldPath)

for f in filesList:
	image = cv2.imread(f)
	if not os.path.exists(newPath + os.path.split(f)[0][len(oldPath):]):
		os.makedirs(newPath + os.path.split(f)[0][len(oldPath):])
	cv2.imwrite(newPath + f[len(oldPath): ], image[image.shape[0] // 4: image.shape[0] // 4 * 3, image.shape[1] // 4: image.shape[1] // 4 * 3])
