# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:28:10 2019

@author: FanXudong
"""

from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc
from tqdm import tqdm
import numpy as np
import os

mnist = input_data.read_data_sets('./MNIST_data/',one_hot=True)

result_path ='./MNIST_data/train/'

def onehot2id(labels):
    return list(labels).index(1)

if not os.path.exists(result_path):
    os.mkdir(result_path)


with open('./MNIST_data/train_labels.txt','w') as labels_txt:
    for i in tqdm(range(len(mnist.train.images))):
        img_vec = mnist.train.images[i,:]
        img_arr = np.reshape(img_vec, [28,28])
        img_lab = mnist.train.labels[i,:]
        img_id = onehot2id(img_lab)
        labels_txt.write(str(i).zfill(6) + '    ' + str(img_id) + '\n')
        img_path = os.path.join(result_path, str(i).zfill(6) + '.jpg')
        misc.imsave(img_path, img_arr)
    
