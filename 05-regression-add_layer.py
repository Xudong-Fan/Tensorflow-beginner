# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:27:06 2019

@author: FanXudong
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font',family='Times New Roman')

tf.set_random_seed(1)
np.random.seed(1)


def add_layer(inputs, in_size, out_size, activation_funiction=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_funiction is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_funiction(Wx_plus_b)
    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#add hidden layer
l1 = add_layer(xs,1,10,activation_funiction=tf.nn.relu)
#add output layer
prediction = add_layer(l1,10,1,activation_funiction=None)

#the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init =tf.initialize_all_variables()



# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

# plot data
plt.scatter(x, y)

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 20, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 1)                     # output layer

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()  # 打开交互模式

for step in range(100):
    # train and net output
    _, current_loss, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.6, 1.1, 'Loss=%.4f' % current_loss, fontdict={'size': 12, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  #关闭交互模式
plt.show()