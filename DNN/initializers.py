import numpy as np

# Initializers By Heuristics
class Random():
    def __init__(self, alpha=0.01):
        # Scale
        self.alpha = alpha  
    
    def __call__(self, shape):
        return np.random.randn(shape[0], shape[1]) * self.alpha
    
class Ones():
    def __call__(self, shape):
        return np.ones(shape)

class Zeros():
    def __call__(self, shape):
        return np.zeros(shape)


# Suitable when activation function used is 'ReLu'
class He():
    """
    n : No. of nodes in previous layer.
    Gaussian Disrtibution : N(0, sqrt(2 / n)).
    """
    def __call__(self, shape):
        return np.random.randn(shape[0], shape[1]) * np.sqrt(2 / shape[1])


# Suitable when activation function used is TanH / Sigmoid
class Xavier():
    """
    n : No. of nodes in previous layer.
    Uniform Distribution : U(-1/sqrt(n), 1/sqrt(n)).
    """

    def __call__(self, shape):
        limit = 1 / np.sqrt(shape[1])
        return np.random.uniform(-limit, limit, shape)

class NormalizedXavier():
    """
    n : No. of nodes in previous layer.
    Uniform Distribution : U(-sqrt(6)/sqrt(n_l-1 + n_l), sqrt(6)/sqrt(n_l-1 + n_l)).
    """
    def __call__(self, shape):
        limit = np.sqrt(6) / np.sqrt(sum(shape))
        return np.random.uniform(-limit, limit, shape)
    

initializers = {
    'Random' : Random,
    'Ones' : Ones,
    'Zeros' : Zeros,
    'He' : He,
    'Xavier' : Xavier,
    'NormalizedXavier' : NormalizedXavier,
}