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

layer1 = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1), name = "Layer1")
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2), name = "Layer2")
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, W3), b3), name = "Layer3")
layerOutput = tf.add(tf.matmul(layer3, W4), b4, name = "LayerOutput")

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
    loss_vec.append(np.sqrt(temp_loss))
    temp_test_loss = sess.run(loss, feed_dict = {x_data: x_vals_test[i:i+1,:],
                                                 y_target: y_vals_test[i:i+1,:]})
    test_loss.append(np.sqrt(temp_test_loss))
    if (i+1)%50 == 0:
        print('Gen: ' + str(i + 1) + '. trainLoss = ' + str(temp_loss)
        + '   testLoss = ' + str(temp_test_loss))
        
tf.train.Saver().save(sess, "./models/model3/model3.ckpt")
tf.summary.FileWriter('./models/model3/logs/', sess.graph)
sess.close()




"""
out:

__main__:6: RuntimeWarning: invalid value encountered in sqrt
__main__:9: RuntimeWarning: invalid value encountered in sqrt
Gen: 50. trainLoss = -298450870000000.0   testLoss = 0.6931472
Gen: 100. trainLoss = 0.6931472   testLoss = 0.6931472
Gen: 150. trainLoss = -1215343100000000.0   testLoss = -242243070000000.0
Gen: 200. trainLoss = -740290760000000.0   testLoss = -497645220000.0
Gen: 250. trainLoss = -1662235400000000.0   testLoss = -796370200000000.0
Gen: 300. trainLoss = 0.6931472   testLoss = -1667026600000000.0
Gen: 350. trainLoss = 0.6931472   testLoss = 0.6931472
Gen: 400. trainLoss = 0.6931472   testLoss = -398178100000000.0
Gen: 450. trainLoss = -3158840800000000.0   testLoss = -3833980000000000.0
Gen: 500. trainLoss = 0.6931472   testLoss = 0.6931472
Gen: 550. trainLoss = -104068390000000.0   testLoss = -518977680000000.0
Gen: 600. trainLoss = 0.6931472   testLoss = 0.6931472
Gen: 650. trainLoss = 0.6931472   testLoss = -8248709600000000.0
Gen: 700. trainLoss = -9357673000000000.0   testLoss = 0.6931472
Gen: 750. trainLoss = -982961850000000.0   testLoss = -4934318000000000.0
Gen: 800. trainLoss = 0.6931472   testLoss = 0.6931472
Traceback (most recent call last):

  File "<ipython-input-6-7427b021dbbd>", line 8, in <module>
    y_target: y_vals_test[i:i+1,:]})

  File "C:\Users\AVSM2\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 877, in run
    run_metadata_ptr)

  File "C:\Users\AVSM2\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1076, in _run
    str(subfeed_t.get_shape())))

ValueError: Cannot feed value of shape (0, 10201) for Tensor 'x-input:0', which has shape '(1, 10201)'
"""