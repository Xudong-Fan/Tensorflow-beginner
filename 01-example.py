# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 00:50:19 2019

@author: FanXudong
"""

# First tensorflow example

import tensorflow as tf
import numpy as np


# Initialize inputs and outputs
x_data = np.random.rand(1000).astype(np.float32) # tensorflow 大部分数据是这个格式
y_data = x_data * 0.1 + 0.3

## create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y_predict = x_data * Weights + biases

loss = tf.reduce_mean(tf.square(y_predict - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
## creat tensorflow structure end 

sess = tf.Session()
sess.run(init)  ## Very important

for epoch in range(201):
    sess.run(train)
    if epoch % 20 == 0:
        print(epoch, sess.run(Weights),sess.run(biases))
        
