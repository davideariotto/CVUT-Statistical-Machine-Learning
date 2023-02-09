import numpy as np

class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        return -np.dot(T,np.log(X).T)[0].reshape(-1,1)

    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        return -T / X


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        :param X: [N]  -- output of last linear layer
        :param T: [N]  -- target labels
        :return: loss [] -- scalar loss
        """
        Xe = np.exp(X - np.max(X, axis=1, keepdims=True))
        Xs = np.sum(Xe, axis=1, keepdims=True)
        softmax = np.divide(Xe, Xs)
        
        return -np.sum(np.log(softmax[T==1])) / X.shape[0]
 

    def delta(self, X, T):
        """
        :param X: [N]  -- output of last linear layer
        :param T: [N]  -- target labels
        :return: gradient in the layer input dx [N]
        (gradient in target T is not needed)
        """
        Xe = np.exp(X - np.max(X, axis=1, keepdims=True))
        Xs = np.sum(Xe, axis=1, keepdims=True)
        softmax = np.divide(Xe, Xs)

        return softmax - T

