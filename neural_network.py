import numpy as np
import random

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden_layer, n_output_layer): #(2, 3, 1) 
        self.n_inputs = n_inputs
        self.n_hidden_layer = n_hidden_layer
        self.n_output_layer = n_output_layer
        self.input_weights = np.random.uniform(low=-1, high=1, size=(self.n_hidden_layer, self.n_inputs)) #A row per neuron in the next hidden layer and a column per input node 
        self.hidden_layers_weights = np.random.uniform(low=-1, high=1, size=(self.n_output_layer, self.n_hidden_layer)) #(1, 3)
        self.bias_hidden = np.zeros((self.n_hidden_layer, 1))
        self.bias_out = np.zeros((self.n_output_layer, 1))
        self.hidden_layer_output = None
        self.lr = 0.1

    def feed_forward(self, X):
        X = np.array(X, ndmin=2).T
        weighted_sum_1 = np.dot(self.input_weights, X) + self.bias_hidden 
        hidden_layer_output = self.sigmoid(weighted_sum_1) 

        weighted_sum_2 = np.dot(self.hidden_layers_weights, hidden_layer_output) + self.bias_out # (1x3)*(3x1) = (1x1)
        output_layer_output = self.sigmoid(weighted_sum_2) 
        
        return output_layer_output

    def fit(self, X, y): 
        X = np.array(X, ndmin=2).T
        y = np.array(y, ndmin=2).T
        #FORWARD PASS
        weighted_sum_1 = np.dot(self.input_weights, X) + self.bias_hidden # (3, 2) * (2, 1) = (3, 1)
        hidden_layer_output = self.sigmoid(weighted_sum_1) #(3, 1)

        weighted_sum_2 = np.dot(self.hidden_layers_weights, hidden_layer_output) + self.bias_out # (1x3)*(3x1) = (1x1)
        output_layer_output = self.sigmoid(weighted_sum_2) 

        #BACKPROP
        #output error = y vector - output result
        output_err = (y - output_layer_output) # (1, 1)
        
        gradients = output_err * self.sigmoid_derivative(output_layer_output)# (1, 1)
        dw = self.lr * np.dot(gradients, hidden_layer_output.T) 

        self.hidden_layers_weights += dw
        self.bias_out += self.lr * output_err

        #Input weights update
        hidden_err = np.dot(self.hidden_layers_weights.T, output_err) #(3, 1) * (1,1) = (3, 1)

        gradients = hidden_err * self.sigmoid_derivative(hidden_layer_output) #(3, 1) x (3, 1)
        dw = np.dot(gradients, X.T) #(3, 1) * (2, 1)

        self.input_weights += self.lr * dw
        self.bias_hidden += self.lr * hidden_err


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

