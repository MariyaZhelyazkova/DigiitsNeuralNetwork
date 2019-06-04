# -*- codi9ng: utf-8 -*-
"""
Created on Thu Apr  4 21:39:30 2019

@author: mimka
"""


import numpy as np

image_size = 28
no_diff_lebels = 10
filepath = "D:/Python/MNIST/"

image_pixel = image_size * image_size
train_data = np.loadtxt(filepath + "mnist_train.csv", delimiter = ",")
test_data = np.loadtxt(filepath + "mnist_test.csv", delimiter = ",")

fac = 0.98 / 255

train_images = np.asfarray(train_data[:,1:]) / 255
test_images = np.asfarray(test_data[:,1:]) / 255

train_labels = np.asfarray(train_data[:,:1])
test_labels = np.asfarray(test_data[:,:1])

lr = range(no_diff_lebels); 
train_lebels_one_hot = (lr==train_labels).astype(float)
test_lebels_one_hot = (lr==test_labels).astype(float)

#Не искаме да има 0 и 1 и в lebel-ите
train_lebels_one_hot[train_lebels_one_hot==0] = 0.01
train_lebels_one_hot[train_lebels_one_hot==1] = 0.99
test_lebels_one_hot[test_lebels_one_hot==0] = 0.01
test_lebels_one_hot[test_lebels_one_hot==1] = 0.99

import pickle
with open(filepath + "pickled_mnist.pkl", "bw") as fh : 
    data = (train_images, 
            test_images, 
            train_labels, 
            test_labels, 
            train_lebels_one_hot, 
            test_lebels_one_hot)
    pickle.dump(data, fh)
    
print("finished")