import numpy as np

class Momentum():
    def __init__(self, momentum):
        self.momentum = momentum
        self.t = 0 # steps taken
    
    def __call__(self, layer, bias=False):
        self.t += 1

        layer.V_w = self.momentum * layer.V_w + (1 - self.momentum) * layer.dW
        layer.V_b = self.momentum * layer.V_b + (1 - self.momentum) * layer.db

        if bias == True:
            layer.V_w /= (1 - self.momentum ** self.t)
            layer.V_b /= (1 - self.momentum ** self.t)
    
    def update(self, layer, lRate):
        layer.W -= lRate * layer.V_w
        layer.b -= lRate * layer.V_b

class RMSprop():
    def __init__(self, beta):
        self.beta = beta
        self.epsilon = 1e-8  # For numerical stablity
        self.t = 0 # Steps taken

    def __call__(self, layer, bias=False):
        self.t += 1

        layer.S_w = self.beta * layer.S_w + (1 - self.beta) * np.power(layer.dW, 2)
        layer.S_b = self.beta * layer.S_b + (1 - self.beta) * np.power(layer.db, 2)

        if bias == True:
            layer.S_w /= (1 - self.beta ** self.t)
            layer.S_b /= (1 - self.beta ** self.t)
    
    def update(self, layer, lRate):
        layer.W -= lRate * layer.dW / np.sqrt(layer.S_w + self.epsilon)
        layer.b -= lRate * layer.db / np.sqrt(layer.S_b + self.epsilon)

class Adam():
    def __init__(self, momentum, beta):
        self.momentum = momentum
        self.beta = beta
        self.epsilon = 1e-8
        self.t = 0 #Steps taken by adam
    
    def __call__(self, layer, bias=True):
        self.t += 1

        layer.V_w = self.momentum * layer.V_w + (1 - self.momentum) * layer.dW
        layer.V_b = self.momentum * layer.V_b + (1 - self.momentum) * layer.db
        layer.S_w = self.beta * layer.S_w + (1 - self.beta) * np.power(layer.dW, 2)
        layer.S_b = self.beta * layer.S_b + (1 - self.beta) * np.power(layer.db, 2)

        if bias == True:
            layer.V_w /= (1 - self.momentum ** self.t)
            layer.V_b /= (1 - self.momentum ** self.t)
            layer.S_w /= (1 - self.beta ** self.t)
            layer.S_b /= (1 - self.beta ** self.t)

    def update(self, layer, lRate):
        layer.W -= lRate * layer.V_w / np.sqrt(layer.S_w + self.epsilon)
        layer.b -= lRate * layer.V_b / np.sqrt(layer.S_b + self.epsilon)

optimizers = {
    'Momentum' : Momentum,
    'RMSprop' : RMSprop,
    'Adam' : Adam,
}

