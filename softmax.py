import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1], [2, 1, 3, 4], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, 3])
nb_class = 3

W = tf.Variable(tf.random_normal([4, nb_class]), name='weight')
b = tf.Variable(tf.random_normal([nb_class]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, 20001):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
            print('step : ', step, ' cost : ', sess.run(cost, feed_dict={x: x_data, y: y_data}))

    all = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1], ]})
    print(all, sess.run(tf.argmax(all, 1)))
