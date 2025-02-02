# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:58:13 2019

@author: mimka
"""

import numpy as np 
@np.vectorize 

def sigmoid(x):
    return 1/(1 + np.e ** -x)

activation_function = sigmoid

from scipy.stats import truncnorm
 
def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10): 
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc = mean, 
                     scale = sd)

class NeuralNetwork: 
    def __init__(self, 
                 num_in_nodes, 
                 num_out_nodes, 
                 num_hidden_nodes, 
                 learning_rate): 
        self.num_in_nodes = num_in_nodes
        self.num_out_nodes = num_out_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrix()
        
    def create_weight_matrix(self):
        rad = 1 / np.sqrt(self.num_in_nodes)
        X = truncated_normal(mean = 0, sd = 1, low = -rad, upp = rad)
        self.wih = X.rvs((self.num_hidden_nodes, self.num_in_nodes))
        
        rad = 1 / np.sqrt(self.num_hidden_nodes)
        X = truncated_normal(mean = 0, sd = 1, low = -rad, upp = rad)
        self.who = X.rvs((self.num_out_nodes, self.num_hidden_nodes))
        
    def train(self, input_vector, target_vector):
        input_vector = np.array(input_vector, ndmin = 2).T
        target_vector = np.array(target_vector, ndmin = 2).T
        
        output_vector1 = np.dot(self.wih, input_vector)
        
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.who, output_hidden)
        
        output_network = activation_function(output_vector2)
        
        #updating the weights
        output_errors = target_vector - output_network
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        
        self.who += tmp
        
        #calculate hidden errors 
        hidden_errors = np.dot(self.who.T, output_errors)
        
        #update the weights
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)
        
    def run(self, input_vector):
        input_vector = np.array(input_vector, ndmin = 2).T
        output_vector = np.dot(self.wih, input_vector)
        
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who, output_vector)
        
        output_vector = activation_function(output_vector)
        
        return output_vector
    
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10,10), int)
        
        for i in range(len(data_array)): 
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm
    
    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum();  
    
    def evaluate(self, data, labels):
        corects, wrongs = 0, 0
        
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            
            if res_max == labels[i]:
                corects += 1
            else:
                wrongs += 1
        return corects, wrongs
    
                