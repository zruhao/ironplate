import tensorflow as tf
import numpy as np

dataPath = './data.csv'
trainSize = 0.5

data = np.genfromtxt(dataPath, delimiter=',')
np.random.shuffle(data)
x_data = data[:, 1:]
y_data = [int(i * 2 - 1.5) for i in data[:, 0]]
dl = int(x_data.shape[0] * trainSize)

x = tf.placeholder(tf.float32, [None, 16])
y = tf.placeholder(tf.float32, [None, 4])
rate = tf.placeholder(tf.float32)

y_onehot = tf.one_hot(indices=y_data, axis=1, depth=4)

W1 = tf.Variable(tf.truncated_normal([16, 1024], mean=0, stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[1024]))

W2 = tf.Variable(tf.truncated_normal([1024, 4], mean=0, stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[4]))

Wx_plus_b1 = tf.matmul(x, W1) + b1
l1 = tf.nn.dropout(tf.nn.relu(Wx_plus_b1), rate=rate)


Wx_plus_b2 = tf.matmul(l1, W2) + b2
prediction = Wx_plus_b2 # tf.nn.softmax(Wx_plus_b2)

loss = tf.losses.softmax_cross_entropy(y, prediction)
train = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(prediction, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	yonehot = sess.run(y_onehot)
	print('epoch	train 			test')
	for j in np.arange(0, 1, 0.1):
		sess.run(tf.global_variables_initializer())
		for i in range(15000):
			sess.run(train, feed_dict={x: x_data[0: dl], y: yonehot[0: dl], rate: 0.7})
			if i == 9999:
				accuTrain = sess.run(accuracy, feed_dict={x: x_data[0: dl], y: yonehot[0: dl], rate: 0})
				accuTest = sess.run(accuracy, feed_dict={x: x_data[dl:], y: yonehot[dl:], rate: 0})
				print(j, '\t', accuTrain, '\t\t', accuTest)