import tensorflow as tf
import numpy as np
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

layer1 = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1), name = "Layer1")
layerOutput = tf.add(tf.matmul(layer1, W2), b2, name = "LayerOutput")

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
    test_temp_loss = sess.run(loss, feed_dict = {x_data: x_vals_train[i:i+1,:],
                                                 y_target: y_vals_train[i:i+1,:]})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%50 == 0:
        print('Gen: ' + str(i + 1) + '. Loss = ' + str(temp_loss))
        
saver = tf.train.Saver()
save_path = saver.save(sess, "./models/model1.ckpt")
file_writer = tf.summary.FileWriter('./models/model1/logs/', sess.graph)
sess.close()