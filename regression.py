import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = [4.0391, 1.3197, 9.5613, 0.5978, 3.5316, 0.1540, 1.6899, 7.3172, 4.5092, 2.9632]
y_data = [11.4215, 10.0112, 30.2991, 1.0625, 13.1776, -3.1976, 6.7367, 23.8550, 14.8951, 11.6137]


W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(1,6001):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if((step %20) ==0):
      print('step:',step,'cost:',cost_val,'W:',sess.run(W),'b:',sess.run(b))

#테스트중입니다