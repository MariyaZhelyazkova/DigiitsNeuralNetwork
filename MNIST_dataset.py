# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:09:10 2019

@author: mimka
"""

import pickle
import numpy as np
import NeuralNetworkClass

with open("D:/Python/MNIST/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

image_size = 28
no_diff_lebels = 10
image_pixel = image_size * image_size


ANN = NeuralNetworkClass.NeuralNetwork(num_in_nodes = image_pixel, 
                                       num_out_nodes = 10, 
                                       num_hidden_nodes = 100, 
                                       learning_rate = 0.1)
print("Network Created")
print("Trainig Begining")

for i in range(len(train_images)):
    ANN.train(train_images[i], train_labels_one_hot[i])

print("Training Completed")

import matplotlib.pyplot as plt

for i in range(20):
    res = ANN.run(test_images[i])
    image = test_images[i].reshape((28,28))
    plt.imshow(image, cmap="Greys")
    plt.show()
    print(test_labels[i], np.argmax(res), np.max(res))