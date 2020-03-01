# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 01:09:56 2019

@author: FanXudong
"""

#import tensorflow as tf
#
#state = tf.Variable(0, name = 'counter')
##print(state.name)
#
#one = tf.constant(1)
#
#new_value = tf.add(state, one)
#update = tf.assign(state, new_value)
#
#init = tf.initialize_all_variables()  # must have if define variables
#
#with tf.Session() as sess:
#    sess.run(init)
#    for _i in range(10):
#        sess.run(update)
#        print(sess.run(state))
        
        
import tensorflow as tf

var = tf.Variable(0)    # our first variable in the "global_variable" set

add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)

with tf.Session() as sess:
    # once define variables, you have to initialize them by doing this
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(var))

        
