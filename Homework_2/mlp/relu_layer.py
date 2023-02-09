import numpy as np

class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        return np.maximum(X, 0)

    def delta(self, Y, delta_next):
        newY = Y > 0    # convert to 1 and 0
        return np.multiply(delta_next, newY)
