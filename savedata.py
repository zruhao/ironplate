import glcm
import numpy as np
import cv2
import getfileslist as gfl

imgsdir = './data/'
datafile = './data.txt'

if imgsdir[-1] != '/':
	imgsdir += '/'

with open(datafile, 'w') as file:
	for s in [str(1.0), str(1.5), str(2.0), str(2.5)]:
		filesList = gfl.getFilesList(imgsdir + s)
		for f in filesList:
			print(f)
			file.write(s)
			img = cv2.imread(f)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			for a, b in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
				mat = glcm.glcm(img_gray, a, b, gray_level=16)
				# print(mat)
				mat = glcm.unif(mat)
				# print(sum(sum(mat)))
				res = glcm.calc(mat)

				for i in res:
					file.write(' ' + str(i))
			file.write('\n')
			break
