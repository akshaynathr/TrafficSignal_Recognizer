#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:10:49 2017

@author: akshaynathr
"""
import tensorflow as tf
import os
import sys
from skimage import data,transform
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def load_data(data_dir):
     directories = [ d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]
     
     labels =[]
     images =[]
     
     for d in directories:
         label_dir = os.path.join(data_dir,d)
         file_names = [os.path.join(label_dir,f) for f in os.listdir(label_dir) if f.endswith('.ppm')]
         
         for f in file_names :
             images.append(data.imread(f))
             labels.append(int(d))
             
     return images, labels
 
    


ROOT_PATH = '/home/akshaynathr/Script/ML/TrafficSignal'

train_data_dir = os.path.join(ROOT_PATH, 'Training')
test_data_dir = os.path.join(ROOT_PATH, 'Testing')


images , labels = load_data(train_data_dir)
print(labels)
images = np.array(images)

print(images)
print(images.size)

print(images[0])

traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))
    
    

# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    

# Show the plot
plt.show()

images28 = [transform.resize(image, (28, 28)) for image in images]


traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images28[traffic_signs[i]].shape, 
                                                  images28[traffic_signs[i]].min(), 
                                                  images28[traffic_signs[i]].max()))
    
images28 = np.array(images28)
images28 = rgb2gray(images28)


traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
    
# Show the plot
plt.show()



#Modelling the neural network

x = tf.placeholder(dtype=tf.float32,shape = [None,28,28]) #images
y = tf.placeholder(dtype=tf.int32,shape=[None])

#Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

#Fullt connected layers
logits = tf.contrib.layers.fully_connected(images_flat,62,tf.nn.relu)

#loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))

#define optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits,1)

#define accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            _ , loss_value = sess.run([train_op,loss], feed_dict={x:images28, y:labels})
            if i %10 ==0:
                print("Loss: ", loss)