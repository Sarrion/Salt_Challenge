# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:00:19 2018

@author: avicent
"""
import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

runfile('C:/Users/AVSM2/Documents/GitHub/Salt_Challenge/target_transformation_to_matrix.py',
        wdir='C:/Users/AVSM2/Documents/GitHub/Salt_Challenge')


sess = tf.Session()

train_indx = np.random.choice(len(images), 
                              round(len(images) * 0.8), replace = False)
test_indx = np.array(list(set(range(len(images))) - set(train_indx)))
x_vals_train = images[train_indx,:]
x_vals_test = images[test_indx,:]
y_vals_train = targets[train_indx,:]
y_vals_test = targets[test_indx,:]

x_data = tf.placeholder(shape = [1,10201], dtype = tf.float32, name = "x-input")
y_target = tf.placeholder(shape = [1,10201], dtype = tf.float32, name = "y-input")

W1 = tf.Variable(tf.random_normal(shape = [10201, 10201], dtype = tf.float32), name = "Weights1")
b1 = tf.Variable(tf.random_normal(shape = [10201], dtype = tf.float32), name = "biases1")
W2 = tf.Variable(tf.random_normal(shape = [10201, 10201], dtype = tf.float32), name = "Weights2")
b2 = tf.Variable(tf.random_normal(shape = [10201], dtype = tf.float32), name = "biases2")
W3 = tf.Variable(tf.random_normal(shape = [10201, 10201], dtype = tf.float32), name = "Weights3")
b3 = tf.Variable(tf.random_normal(shape = [10201], dtype = tf.float32), name = "biases3")
W4 = tf.Variable(tf.random_normal(shape = [10201, 10201], dtype = tf.float32), name = "Weights4")
b4 = tf.Variable(tf.random_normal(shape = [10201], dtype = tf.float32), name = "biases4")
W5 = tf.Variable(tf.random_normal(shape = [10201, 10201], dtype = tf.float32), name = "Weights5")
b5 = tf.Variable(tf.random_normal(shape = [10201], dtype = tf.float32), name = "biases5")


layer1 = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1), name = "Layer1")
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2), name = "Layer2")
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, W3), b3), name = "Layer3")
layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, W4), b4), name = "Layer4")
layerOutput = tf.add(tf.matmul(layer4, W5), b5, name = "LayerOutput")

loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(layerOutput, y_target))   
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001,
                                   epsilon = 0.1)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
test_loss = []
for i in range(len(x_vals_train)):
    sess.run(train_step, feed_dict = {x_data: x_vals_train[i:i+1,:],
                                      y_target: y_vals_train[i:i+1,:]})
    temp_loss = sess.run(loss, feed_dict = {x_data: x_vals_train[i:i+1,:],
                                            y_target: y_vals_train[i:i+1,:]})
    
# =============================================================================
#     loss_vec.append(np.sqrt(temp_loss))
#     temp_test_loss = sess.run(loss, feed_dict = {x_data: x_vals_test[i:i+1,:],
#                                                  y_target: y_vals_test[i:i+1,:]})
#     test_loss.append(np.sqrt(temp_test_loss))
# =============================================================================
    if (i+1)%50 == 0:
        print('Gen: ' + str(i + 1) + '. trainLoss = ' + str(temp_loss))
#        + '   testLoss = ' + str(temp_test_loss))
        
tf.train.Saver().save(sess, "./models/model4/model4.ckpt")
tf.summary.FileWriter('./models/model4/logs/', sess.graph)
sess.close()