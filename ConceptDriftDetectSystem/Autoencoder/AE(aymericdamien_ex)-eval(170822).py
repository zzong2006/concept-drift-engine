# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True, validation_size = 100)

#print(mnist.train.labels[1]) # mnist data print
#print(mnist.train.images[1]) # mnist data print
'''
for i in range(12):
    for j in range(784):
        print(mnist.train.images[i][j], end=",")
    print("") 
'''

# Parameters
display_step = 1
FILE_OUTPUT = True
examples_to_vectorize = 2519965

examples_to_show = [0,1,2,3,4,5,6,7,8,9,10,11] # default

# Network Parameters
n_hidden_1 = 64 # 1st layer num features
n_hidden_2 = 25 # 2nd layer num features
n_input = 1440 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

'''
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
'''

# Initializing the variables
init = tf.global_variables_initializer()

# Model Saver op
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    ckpt = tf.train.get_checkpoint_state("./trained_AE")
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
    '''
    # Applying encode and decode over test set
    encode = sess.run(
        encoder_op, feed_dict={X: mnist.test.images[:examples_to_show]})
    #print(encode[0])
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    '''
    
    if FILE_OUTPUT == True:
        # Applying encode and decode over test set
        encode = sess.run(
            encoder_op, feed_dict={X: mnist.test.images[:examples_to_vectorize]})
        #print(encode[0])
        encode_decode = sess.run(
            y_pred, feed_dict={X: mnist.test.images[:examples_to_vectorize]})
    
        out_encoded_vector = open("./encoded_data", "w")
        for i in range(examples_to_vectorize):
            # print vectors to file
            for j in range(n_hidden_2):
                if(j == (n_hidden_2 - 1)):
                    out_encoded_vector.write(str(encode[i][j]))
                else:
                    out_encoded_vector.write(str(encode[i][j]) + ",")
            out_encoded_vector.write("\n")
        out_encoded_vector.close()
    
    else:
        # Applying encode and decode over test set
        encode = sess.run(
            encoder_op, feed_dict={X: mnist.test.images[:examples_to_vectorize]})
        #print(encode[0])
        encode_decode = sess.run(
            y_pred, feed_dict={X: mnist.test.images[:examples_to_vectorize]})
        
        # Compare original images with their reconstructions
        f, a = plt.subplots(3, 12, figsize=(12, 3))
        
        z = 0;
        for i in examples_to_show:
            a[0][z].imshow(np.reshape(mnist.test.images[i], (40, 36)), clim=(0.0,1.0))
            a[1][z].imshow(np.reshape(encode[i], (5, 5)), clim=(0.0,1.0))
            # print vectors to standard output
            for j in range(n_hidden_2):
                print(encode[i][j], end=" ")
            print("")
            a[2][z].imshow(np.reshape(encode_decode[i], (40, 36)), clim=(0.0,1.0))
            z = z + 1;
        f.show()
        plt.draw()
        plt.waitforbuttonpress()
