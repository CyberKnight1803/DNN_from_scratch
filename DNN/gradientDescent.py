import numpy as np
from optimizers import optimizers

class GD():
    def __init__(self, lRate=0.45):
        self.lRate = lRate
    
    def update(self, layers):
        for layer in layers:
            layer.W -= self.lRate * layer.dW
            layer.b -= self.lRate * layer.db

class BatchGD(GD):
    def __init__(self, lRate=0.45, momentum=0, optimizer=None):
        super().__init__(lRate=lRate)
        self.optimizer = optimizer
        if optimizer == 'Momentum':
            self.optimizer = optimizers[optimizer](momentum)

    
    def __call__(self, X, y, layers, mechanism, costs, itr, print_cost=True):

        AL, caches = mechanism['forward_prop'](X)
        cost = mechanism['compute_cost'](y, AL)
        mechanism['backward_prop'](AL, y, caches)
        costs.append(cost)

        if self.optimizer == None:
            self.update(layers)
        else:
            for layer in layers:
                self.optimizer(layer)
                self.optimizer.update(layer, self.lRate)

        if print_cost and itr % 100 == 0:
            print(f"Cost after iteration{itr}: {cost}")


class StochasticGD(GD):
    def __init__(self, lRate=0.45, momentum=0, beta=0, optimizer=None):
        super().__init__(lRate=lRate)
        self.momentum = momentum

        if optimizer == 'Momentum':
            self.optimizer = optimizers[optimizer](momentum)
        elif optimizer == 'RMSprop':
            self.optimizer = optimizers[optimizer](beta)
        elif optimizer == 'Adam':
            self.optimizer = optimizers[optimizer](momentum, beta)

    
    def update(self, layers):
        for layer in layers:
            self.optimizer(layer)
            self.optimizer.update(layer, self.lRate)
    
    def __call__(self, X, y, layers, mechanism, costs, itr, print_cost=True):
        m = X.shape[1]  
        cost = 0
        for i in range(0, m):
            AL, caches = mechanism['forward_prop'](X[:, i].reshape(-1, 1))
            cost = mechanism['compute_cost'](y[:, i].reshape(1, -1), AL)
            mechanism['backward_prop'](AL, y[:, i].reshape(1, -1), caches)
            self.update(layers, itr)
        
        costs.append(cost)
        if print_cost and itr % 10 == 0:
            print(f"Cost after iteration{itr}: {cost}")
            

class MiniBatchGD(GD):
    def __init__(self, lRate=0.45, momentum=0.9, beta=0.98, batch_size=1024, optimizer=None):
        super().__init__(lRate=lRate)
        self.batch_size = batch_size
        self.momentum = momentum
        self.optimizer = optimizer
        if optimizer == 'Momentum':
            self.optimizer = optimizers[optimizer](momentum)
        elif optimizer == 'RMSprop':
            self.optimizer = optimizers[optimizer](beta)
        elif optimizer == 'Adam':
            self.optimizer = optimizers[optimizer](momentum, beta)
            

    def update(self, layers):
        for layer in layers:
            self.optimizer(layer)
            self.optimizer.update(layer, self.lRate)
    
    def createBatches(self, X, y, seed=0):

        m = X.shape[1]
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_y = y[:, permutation].reshape(1, m)

        batches = []
        nBatches = int(m / self.batch_size)

        for i in range(0, nBatches):
            batch_X = shuffled_X[:, i * self.batch_size : (i + 1) * self.batch_size]
            batch_y = shuffled_y[:, i * self.batch_size : (i + 1) * self.batch_size]

            miniBatch = (batch_X, batch_y)
            batches.append(miniBatch)
        
        # Handling last batch 
        if m % self.batch_size != 0:
            batch_X = shuffled_X[:, nBatches * self.batch_size :]
            batch_y = shuffled_y[:, nBatches * self.batch_size :]

            miniBatch = (batch_X, batch_y)
            batches.append(miniBatch)
        
        return batches
    
    def __call__(self, X, y, layers, mechanism, costs, itr, print_cost=True):

        batches = self.createBatches(X, y)
        cost = 0
        for b in range(0, len(batches)):
            _X, _y = batches[b]

            AL, caches = mechanism['forward_prop'](_X)
            cost = mechanism['compute_cost'](_y, AL)
            mechanism['backward_prop'](AL, _y, caches)

            self.update(layers)
        
        costs.append(cost)
        if print_cost and itr % 10 == 0:
            print(f"Cost after iteration{itr}: {cost}")


GD_variants = {
    'BatchGD' : BatchGD,
    'StochasticGD' : StochasticGD,
    'MiniBatchGD' : MiniBatchGD,
}