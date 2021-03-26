import numpy as np

class CrossEntropy():

    def __call__(self, y, A):

        # Preventive measure to avoid division by zero
        P = np.clip(A, 1e-15, 1 - 1e-15) 
        Loss = np.multiply(y, np.log(P)) + np.multiply((1 - y), np.log(1 - P))
        return Loss

    def derivative(self, y, A):

        # To avoid exploding gradients
        P = np.clip(A, 1e-15, 1 - 1e-15)
        dA = - (np.divide(y, P) - np.divide(1 - y, 1 - P))
        
        return dA

class SquareLoss():

    def __call__(self, y, A):
        loss = 0.5 * np.power((y - A), 2)
        return loss

    def derivative(self, y, A):
        dA = -(y - A)
        return dA

loss_functions = {
    'CrossEntropy' : CrossEntropy,
    'SquareLoss' : SquareLoss,
}