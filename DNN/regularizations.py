import numpy as np

class L2():
    def __init__(self, gamma=0.7):
        # Lambda paramter in regularization
        self.gamma = gamma

    def __call__(self, layers, m):
        self.sigma = 0
        for layer in layers:
            self.sigma += np.sum(np.square(layer.W))

        return (1 / m) * (self.gamma / 2) * self.sigma
    
    def back_prop(self, W, m):
        return (self.gamma / m) * W

class Dropout():
    def __init__(self, keep_probs=0.5):
        self.keep_probs = keep_probs

    def __call__(self):
        pass

regularizers = {
    'L2' : L2,
    'Dropout' : Dropout,
}

