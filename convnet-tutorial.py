
# coding: utf-8

# Some consideration before starting:
# -Edit cifar10.py file, setting the directory where CIFAR10 dataset is located.

# In[1]:

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import tensorflow as tf

from cifar10 import CIFAR10


# In[2]:

sess = tf.InteractiveSession()


# In[3]:

# Load dataset
cifar10 = CIFAR10(batch_size=100, validation_proportion=0.1)


# In[4]:

# Model blocks
def conv_layer(input_tensor, kernel_shape):
    # input_tensor b01c
    # kernel_shape 01-in-out
    weights = tf.get_variable("weights", kernel_shape,
                               initializer = tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases", [kernel_shape[3]],
                             initializer=tf.constant_initializer(0.0))
    # Other options are to use He et. al init. for weights and 0.01 
    # to init. biases.
    conv = tf.nn.conv2d(input_tensor, weights, 
                       strides = [1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def fc_layer(input_tensor, weights_shape):
    # weights_shape in-out
    weights = tf.get_variable("weights", weights_shape,
                              initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", [weights_shape[1]],
                             initializer=tf.constant_initializer(0.0))
    mult_out = tf.matmul(input_tensor, weights)
    return tf.nn.relu(mult_out+biases)
    


# In[5]:

# Model
model_input = tf.placeholder(tf.float32, name='model_input')

keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

target = tf.placeholder(tf.float32, name='target')

with tf.variable_scope('conv1'):
    conv1_out = conv_layer(model_input, [5, 5, 3, 64])

pool1_out = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME',
                          name='pool1')

with tf.variable_scope('conv2'):
    conv2_out = conv_layer(pool1_out, [5, 5, 64, 64])
    
pool2_out = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME',
                          name='pool2')

pool2_out_flat = tf.reshape(pool2_out, [-1, 8*8*64], name='pool2_flat')
with tf.variable_scope('fc1'):
    fc1_out = fc_layer(pool2_out_flat, [8*8*64, 512])

fc1_out_drop = tf.nn.dropout(fc1_out, keep_prob)

with tf.variable_scope('fc2'):
    fc2_out = fc_layer(fc1_out_drop, [512, 10])
    
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(fc2_out, target,
                                           name='cross_entropy'))


# In[6]:

# Optimization
optimizer = tf.train.AdamOptimizer(1e-4)
grads_vars = optimizer.compute_gradients(cross_entropy)
optimizer.apply_gradients(grads_vars)
train_step = optimizer.minimize(cross_entropy)


# Metrics
correct_prediction = tf.equal(tf.argmax(fc2_out, 1),
                             tf.argmax(target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Useful training functions
def validate():
    data, labels = cifar10.getValidationSet()
    return accuracy.eval(feed_dict={
            model_input: data,
            target: labels,
            keep_prob: 1.0
        })
def test():
    data, labels = cifar10.getTestSet()
    return accuracy.eval(feed_dict={
            model_input: data,
            target: labels,
            keep_prob: 1.0
        })
    


# In[7]:

# Train model
sess.run(tf.initialize_all_variables())
cifar10.reset()
print "Trainable variables"
for n in tf.trainable_variables():
    print n.name

epochs = 3
mean_gradients = np.zeros([len(tf.trainable_variables()),epochs])
std_gradients = np.zeros([len(tf.trainable_variables()),epochs])

n_batches = cifar10.n_batches
while cifar10.getEpoch()<epochs:
    epoch = cifar10.getEpoch()
    batch, batch_idx = cifar10.nextBatch()
    batch_data = batch[0]
    batch_labels = batch[1]
    _, loss, grads = sess.run((train_step, cross_entropy, grads_vars), 
                      feed_dict={
            model_input: batch_data,
            target: batch_labels,
            keep_prob: 0.5
        })
    for layer in range(len(tf.trainable_variables())):
        mean_gradients[layer,epoch] = np.mean(np.abs(grads[layer][0]))
        std_gradients[layer,epoch] = np.std(np.abs(grads[layer][0]))
    #for elem in grads[0]:
    #    print elem.shape
        
    if batch_idx==0:
        print "Epoch %d, loss %f" %(epoch, loss)
        validation_accuracy = validate()
        print "Validation accuracy %f"%(validation_accuracy)
        test_accuracy = test()
        print "Test_accuracy %f"%(test_accuracy)
        


# In[8]:

x = np.arange(epochs)
plt.errorbar(x,mean_gradients[0,:],std_gradients[0,:])
plt.hold(True)
plt.errorbar(x,mean_gradients[2,:],std_gradients[2,:])
plt.errorbar(x,mean_gradients[4,:],std_gradients[4,:])
plt.errorbar(x,mean_gradients[6,:],std_gradients[6,:])
plt.ylabel('Gradient')
plt.xlabel('Epochs')
plt.title('Weights Gradient by Layer')
plt.legend(["conv1", "conv2","fc1","fc2"])
plt.hold(False)
plt.show()
plt.errorbar(x,mean_gradients[1,:],std_gradients[1,:])
plt.hold(True)
plt.errorbar(x,mean_gradients[3,:],std_gradients[3,:])
plt.errorbar(x,mean_gradients[5,:],std_gradients[5,:])
plt.errorbar(x,mean_gradients[7,:],std_gradients[7,:])
plt.ylabel('Gradient')
plt.xlabel('Epochs')
plt.title('Biases Gradient by Layer')
plt.legend(["conv1", "conv2","fc1","fc2"])
plt.hold(False)
plt.show()


# In[ ]:



