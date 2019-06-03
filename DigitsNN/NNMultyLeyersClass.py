# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:19:06 2019

@author: mimka
"""

import numpy  as np
from scipy.special import expit as activation_function 
from scipy.stats import truncnorm 

def truncated_normal(mean = 0, sd = 1, low = 0, up = 10): 
    return truncnorm((low - mean) / sd, 
                     (up - mean) / sd, 
                     loc = mean, 
                     scale = sd)

class NeuralNetwork: 
    def __init__(self, network_structure, learning_rate, bias = None): 
        self.structure = network_structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.layers_num = len(self.structure)
        self.create_weight_matrixes()
        
    def create_weight_matrixes(self):        
        bias_node = 1 if self.bias else 0
        
        self.weigth_matrixes = []
        
        layer_index = 1
        
        while layer_index < self.layers_num: 
            #previous layer is  the input 
            in_nodes = self.structure[layer_index -1]
            #current layer is the output
            out_nodes = self.structure[layer_index]
            n = (in_nodes + bias_node) * out_nodes
            rad = 1 / np.sqrt(in_nodes)
            X = truncated_normal(mean = 2, sd = 1, low = -rad, up = rad)
            wm = X.rvs(n).reshape((out_nodes, in_nodes + bias_node))
            self.weigth_matrixes.append(wm)
            layer_index += 1
            
    def train_single(self, input_vector, target_vector):
        input_vector = np.array(input_vector, ndmin = 2).T
        result_vectors = [input_vector]
        layer_index = 0
        
        while layer_index < self.layers_num - 1: 
            in_vector = result_vectors[-1] #always gets the last item
            if self.bias: 
                in_vector = np.concatenate(in_vector, [[self.bias]])
                result_vectors[-1] = in_vector
            x = np.dot(self.weigth_matrixes[layer_index], in_vector) 
            out_vector = activation_function(x)
            result_vectors.append(out_vector)
            layer_index += 1
        
        
        layer_index = self.layers_num - 1
        target_vector = np.array(target_vector, ndmin = 2).T
        output_error = target_vector - out_vector
        
        while layer_index > 0: 
            out_vector = result_vectors[layer_index]
            in_vector = result_vectors[layer_index -1]
            if self.bias and not layer_index == (self.layers_num -1): 
                out_vector = out_vector[:-1,:].copy()
            tmp = output_error * out_vector  * (1.0 - out_vector)
            tmp = np.dot(tmp, in_vector.T)
            
            self.weigth_matrixes[layer_index - 1] += self.learning_rate * tmp
            
            output_error = np.dot(self.weigth_matrixes[layer_index - 1].T, 
                                  output_error)
            if self.bias: 
                output_error = output_error[:-1,:]
                
            layer_index -= 1
    
    def train(self, data_array, 
              labels_one_hot_array, 
              epochs = 1, 
              intermedieate_result = False):
        intermediate_weights = []
        for epoch in range(epochs): 
            for i in range(len(data_array)): 
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermedieate_result: 
                intermediate_weights.append((self.wih.copy(), self.who.copy()))
                
        return intermediate_weights
    
    def run(self, input_vector): 
        if self.bias: 
            input_vector = np.concatenate( (input_vector, [self.bias]) )
        in_vector = np.array(input_vector, ndmin = 2).T
        layer_index = 1
        while layer_index < self.layers_num: 
            x = np.dot(self.weigth_matrixes[layer_index - 1], in_vector)
            out_vector = activation_function(x)
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate( (in_vector, [[self.bias]]) )
            layer_index += 1
            
        return out_vector
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)): 
            res = self.run(data[i])
            res_max =res.argmax()
            if res_max == labels[i]: 
                corrects += 1
            else: 
                wrongs += 1
        return corrects, wrongs