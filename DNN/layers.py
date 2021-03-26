import numpy as np
from activation_functions import activations
from initializers import initializers
from regularizations import regularizers

class Layer():
    def __init__(self, l, layer_shape, activation='ReLu', initializer='He', regularizer=None):

        self.l = l
        self.activation = activations[activation]()
        self.initializer = initializers[initializer]()
        self.regularizer = regularizer
    
        #Parameters
        self.W = self.initializer(layer_shape)
        self.b = np.zeros((layer_shape[0], 1))

        # Used when momentum is set.
        self.V_w = np.zeros(np.shape(self.W)) 
        self.V_b = np.zeros(np.shape(self.b))

        #Used when RMSprop is set
        self.S_w = np.zeros(np.shape(self.W))
        self.S_b = np.zeros(np.shape(self.b))
    
    def __str__(self):
        S = 'Layer ' + str(self.l) + ' W shape : ' + str(self.W.shape), 'b shape : ' + str(self.b.shape)
        return S
    
    def forward_pass(self, _A):

        Z = np.dot(self.W, _A) + self.b
        A = self.activation(Z)

        return A, Z
    
    def backward_pass(self, dA, cache):
        _A, Z = cache
        m = _A.shape[1]

        dZ = dA * self.activation.derivative(Z)

        self.dW = np.dot(dZ, _A.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m

        if self.regularizer != None:
            self.dW += self.regularizer.back_prop(self.W, m)

        _dA = np.dot(self.W.T, dZ)

        return _dA

    
