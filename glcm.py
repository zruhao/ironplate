import cv2
import numpy as np

# 计算灰度共生矩阵
def glcm(img_gray, a, b, gray_level=None):
	img_gray = img_gray.astype(np.float32)
	n, m = img_gray.shape
	max_gray = img_gray.max()
	if gray_level == None:
		gray_level = max_gray + 1
	else:
		img_gray = img_gray * (gray_level - 1) // max_gray
	mat = np.zeros((gray_level, gray_level))
	for i in range(n):
		for j in range(m):
			if 0 <= i + a < n and 0 <= j + b < m:
				mat[int(img_gray[i, j]), int(img_gray[i + a, j + b])] += 1

	return mat

# 归一化
def unif(mat):
	return mat / mat.sum()

# 计算 ASM ENT CON IDM
def calc(mat):
	n, m = mat.shape
	asm = 0
	ent = 0
	con = 0
	idm = 0
	for i in range(n):
		for j in range(m):
			asm += mat[i, j] ** 2
			ent -= mat[i, j] * np.log(mat[i, j]) if mat[i, j] > 0 else 0
			con += ((i - j) ** 2) * mat[i, j]
			idm += mat[i, j] / (1 + (i - j) ** 2)
	return asm, ent, con, idm

# if __name__ == '__main__':
	# fps = ['./new/1.0/IMG20190614092852.jpg', 
	# 		'./new/1.5/IMG20190614101044.jpg', 
	# 		'./new/2.0/IMG20190614105601.jpg', 
	# 		'./new/2.5/IMG20190614124857.jpg']
	# # import getfileslist as gfl
	# # fps = gfl.getFilesList('./new/2.5/')
	# for fp in fps:
	# 	img = cv2.imread(fp)
	# 	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 	# print(img_gray)
	# 	for a, b in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
	# 		mat = glcm(img_gray, a, b, gray_level=16)
	# 		# print(mat)
	# 		mat = unif(mat)
	# 		# print(sum(sum(mat)))
	# 		print(calc(mat))
