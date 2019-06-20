import os

def getFilesList(path):
	def getFilesPath(path, filesPathList):
		for f in os.listdir(path):
			if os.path.isfile(path + f):
				filesPathList.append(path + f)
			else:
				getFilesPath(path + f + '/', filesPathList)

	filesPathList = []
	if path[-1] != '/':
		path += '/'
	getFilesPath(path, filesPathList)
	return filesPathList

if __name__ == '__main__':
	print(getFilesList('./data'))