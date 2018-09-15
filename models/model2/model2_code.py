runfile('C:/Users/AVSM2/Documents/GitHub/Salt_Challenge/target_transformation_to_matrix.py',
        wdir='C:/Users/AVSM2/Documents/GitHub/Salt_Challenge')

import tensorflow as tf
#import numpy as np
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

W1 = tf.Variable(tf.random_normal(shape = [10201, 10201]), name = "Weights1")
b1 = tf.Variable(tf.random_normal(shape = [10201]), name = "biases1")
W2 = tf.Variable(tf.random_normal(shape = [10201, 10201]), name = "Weights2")
b2 = tf.Variable(tf.random_normal(shape = [10201]), name = "biases2")
W3 = tf.Variable(tf.random_normal(shape = [10201, 10201]), name = "Weights3")
b3 = tf.Variable(tf.random_normal(shape = [10201]), name = "biases3")

layer1 = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1), name = "Layer1")
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2), name = "Layer2")
layerOutput = tf.add(tf.matmul(layer2, W3), b3, name = "LayerOutput")

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
    test_loss = sess.run(loss, feed_dict = {x_data: x_vals_test[i:i+1,:],
                                            y_target: y_vals_test[i:i+1,:]})
    test_loss.append(np.sqrt(test_loss))
    if (i+1)%50 == 0:
        print('Gen: ' + str(i + 1) + '. trainLoss = ' + str(temp_loss)
        + '.   testLoss = ' + str(test_loss))
        
tf.train.Saver().save(sess, "./models/model2/model2.ckpt")
tf.summary.FileWriter('./models/model2/logs/', sess.graph)
sess.close()





"""
out(NOTE: There has been a problem with the testLoss output, is the same that trainLoss due to an error in the code):

__main__:42: RuntimeWarning: invalid value encountered in sqrt
__main__:45: RuntimeWarning: invalid value encountered in sqrt
Gen: 50. trainLoss = -258243.19.   testLoss = -258243.19
Gen: 100. trainLoss = -593533.8.   testLoss = -593533.8
Gen: 150. trainLoss = -712623.25.   testLoss = -712623.25
Gen: 200. trainLoss = -139909950.0.   testLoss = -139909950.0
Gen: 250. trainLoss = -583169000.0.   testLoss = -583169000.0
Gen: 300. trainLoss = -812745600.0.   testLoss = -812745600.0
Gen: 350. trainLoss = -3289492700.0.   testLoss = -3289492700.0
Gen: 400. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 450. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 500. trainLoss = -3168789200.0.   testLoss = -3168789200.0
Gen: 550. trainLoss = -14007367000.0.   testLoss = -14007367000.0
Gen: 600. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 650. trainLoss = -25188764000.0.   testLoss = -25188764000.0
Gen: 700. trainLoss = -3476224.8.   testLoss = -3476224.8
Gen: 750. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 800. trainLoss = -65033613000.0.   testLoss = -65033613000.0
Gen: 850. trainLoss = -66138542000.0.   testLoss = -66138542000.0
Gen: 900. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 950. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1000. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1050. trainLoss = -2902898700.0.   testLoss = -2902898700.0
Gen: 1100. trainLoss = -101622030000.0.   testLoss = -101622030000.0
Gen: 1150. trainLoss = -10574642000.0.   testLoss = -10574642000.0
Gen: 1200. trainLoss = -2355053000.0.   testLoss = -2355053000.0
Gen: 1250. trainLoss = -242092150000.0.   testLoss = -242092150000.0
Gen: 1300. trainLoss = -73872584.0.   testLoss = -73872584.0
Gen: 1350. trainLoss = -87486600000.0.   testLoss = -87486600000.0
Gen: 1400. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1450. trainLoss = -81801960000.0.   testLoss = -81801960000.0
Gen: 1500. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1550. trainLoss = -199053590000.0.   testLoss = -199053590000.0
Gen: 1600. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1650. trainLoss = -209108320000.0.   testLoss = -209108320000.0
Gen: 1700. trainLoss = -289954460000.0.   testLoss = -289954460000.0
Gen: 1750. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1800. trainLoss = -168054640000.0.   testLoss = -168054640000.0
Gen: 1850. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1900. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 1950. trainLoss = -82290500000.0.   testLoss = -82290500000.0
Gen: 2000. trainLoss = -124126630000.0.   testLoss = -124126630000.0
Gen: 2050. trainLoss = -316211000000.0.   testLoss = -316211000000.0
Gen: 2100. trainLoss = -192502070000.0.   testLoss = -192502070000.0
Gen: 2150. trainLoss = -1064779800000.0.   testLoss = -1064779800000.0
Gen: 2200. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 2250. trainLoss = -534930550000.0.   testLoss = -534930550000.0
Gen: 2300. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 2350. trainLoss = -1725246500000.0.   testLoss = -1725246500000.0
Gen: 2400. trainLoss = -988331500000.0.   testLoss = -988331500000.0
Gen: 2450. trainLoss = -2256583500000.0.   testLoss = -2256583500000.0
Gen: 2500. trainLoss = -13038419000.0.   testLoss = -13038419000.0
Gen: 2550. trainLoss = -523802540000.0.   testLoss = -523802540000.0
Gen: 2600. trainLoss = -1147300500000.0.   testLoss = -1147300500000.0
Gen: 2650. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 2700. trainLoss = -17122714000.0.   testLoss = -17122714000.0
Gen: 2750. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 2800. trainLoss = -2527797400000.0.   testLoss = -2527797400000.0
Gen: 2850. trainLoss = -2049328200000.0.   testLoss = -2049328200000.0
Gen: 2900. trainLoss = -135121660000.0.   testLoss = -135121660000.0
Gen: 2950. trainLoss = -411856450.0.   testLoss = -411856450.0
Gen: 3000. trainLoss = 0.6931472.   testLoss = 0.6931472
Gen: 3050. trainLoss = -3393471000000.0.   testLoss = -3393471000000.0
Gen: 3100. trainLoss = -610903500000.0.   testLoss = -610903500000.0
Gen: 3150. trainLoss = -1582720600000.0.   testLoss = -1582720600000.0
Gen: 3200. trainLoss = 0.6931472.   testLoss = 0.6931472
"""