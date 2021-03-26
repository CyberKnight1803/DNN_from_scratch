import numpy as np
from layers import Layer
from loss_functions import CrossEntropy
from gradientDescent import GD_variants
from regularizations import regularizers


class DNN():

    def __init__(self, layer_dims, momentum=0, beta=0, lRate=0.45, n_iters=5000, activation='ReLu', initializer = 'He', GD_type='StochasticGD',
    batch_size=None, optimizer=None, regularizer=None, regularizer_const=0):
        self.lRate = lRate
        self.n_iters = n_iters
        self.loss = CrossEntropy()
        self.optimizer = optimizer

        self.regularizer = regularizer
        if regularizer != None:
            self.regularizer = regularizers[regularizer](regularizer_const)
        
        self.GD_type = None
        if GD_type == 'BatchGD':
            self.GD_type = GD_variants[GD_type](self.lRate)
        elif GD_type == 'StochasticGD':
            self.GD_type = GD_variants[GD_type](self.lRate, momentum, beta, self.optimizer)
        elif GD_type == 'MiniBatchGD':
            self.GD_type = GD_variants[GD_type](self.lRate, momentum, beta, batch_size, self.optimizer)

        self.layers = []
        self.n_layers = len(layer_dims) - 1

        #Initializing all layers with ReLu except last
        for l in range(1, self.n_layers):
            layer_shape = (layer_dims[l], layer_dims[l - 1])
            self.layers.append(Layer(l, layer_shape, activation, initializer, self.regularizer))
            print(self.layers[l - 1].__str__())
        
        layer_shape = (layer_dims[self.n_layers], layer_dims[self.n_layers - 1]) 
        self.layers.append(Layer(self.n_layers, layer_shape, 'Sigmoid', 'Random', self.regularizer))
        print(self.layers[self.n_layers - 1].__str__())

    def forward_propagation(self, X):
        A = X
        caches = []

        for layer in self.layers:
            _A = A
            A, Z = layer.forward_pass(_A)

            caches.append((_A, Z))

        return A, caches

    def compute_cost(self, y, AL):

        J = - np.sum(self.loss(y, AL)) / self.m
        if self.regularizer != None:
            J += self.regularizer(self.layers, self.m)
            
        return J

    def backward_propagation(self, AL, y, caches):
        dAL = self.loss.derivative(y, AL)

        _dA = dAL
        for l in reversed(range(self.n_layers)):
            dA = _dA
            _dA = self.layers[l].backward_pass(dA, caches[l])

    def fit(self, X, y, print_cost=False):
        self.m = X.shape[1]
        self.costs = []
        mechanism = {
            'forward_prop' : self.forward_propagation,
            'backward_prop' : self.backward_propagation,
            'compute_cost' : self.compute_cost
        }

        for i in range(0, self.n_iters):
            self.GD_type(X, y, self.layers, mechanism, self.costs, i, print_cost=True)

        return self.costs
    
    def performance(self, X, y):
        A, caches = self.forward_propagation(X)
        P = np.around(A)
        metrics = {}
        
        correctPred = P[P == y]
        trueP = np.sum(correctPred == 1)

        precision = trueP / np.sum(P == 1)
        recall = trueP / np.sum(y == 1)
        F1Score = 2 * (precision * recall) / (precision + recall)

        acc = np.sum(P == y) / y.shape[1]

        metrics = { 'accuracy' : acc,
                    'precision' : precision,
                    'recall' : recall,
                    'F1Score' : F1Score}

        return metrics

