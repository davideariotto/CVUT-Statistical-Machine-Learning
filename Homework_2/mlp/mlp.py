import numpy as np
import matplotlib.pyplot as plt

class MLP(object):
    def __init__(self, n_inputs, layers, loss, output_layers=[]):
        """
        MLP constructor.
        :param n_inputs:
        :param layers: list of layers
        :param loss: loss function layer
        :param output_layers: list of layers appended to "layers" in evaluation phase, parameters of these are not used
        in training phase
        """
        self.n_inputs = n_inputs
        self.layers = layers
        self.output_layers = output_layers
        self.loss = loss
        self.first_param_layer = layers[-1]
        for l in layers:
            if l.has_params():
                self.first_param_layer = l
                break

    def propagate(self, X, output_layers=True, last_layer=None):
        """
        Feedforwad network propagation
        :param X: input data, shape (n_samples, n_inputs)
        :param output_layers: controls whether the self.output_layers are appended to the self.layers in evaluatin
        :param last_layer: if not None, the propagation will stop at layer with this name
        :return: propagated inputs, shape (n_samples, n_units_of_the_last_layer)
        """
        layers = self.layers + (self.output_layers if output_layers else [])
        if last_layer is not None:
            assert isinstance(last_layer, str)
            layer_names = [layer.name for layer in layers]
            layers = layers[0: layer_names.index(last_layer) + 1]
        for layer in layers:
            X = layer.forward(X)
        return X

    def evaluate(self, X, T):
        """
        Computes loss.
        :param X: input data, shape (n_samples, n_inputs)
        :param T: target labels, shape (n_samples, n_outputs)
        :return:
        """
        return self.loss.forward(self.propagate(X, output_layers=False), T)

    def gradient(self, X, T):
        """
        Computes gradient of loss w.r.t. all network parameters.
        :param X: input data, shape (n_samples, n_inputs)
        :param T: target labels, shape (n_samples, n_outputs)
        :return: a dict of records in which key is the layer.name and value the output of grad function
        """
        # manually iterate over layers saving forward pass
        outputs = []
        inputs = []
        for l in self.layers:
            inputs.append(X)
            X = l.forward(X)
            outputs.append(X)
        dL = self.loss.delta(X,T)
        res = {}
        # Loop over the layers in reverse order for backpropagation
        for i,l in zip(range(len(self.layers)-1,-1,-1),self.layers[::-1]):
            if l.has_params():
                # Compute the gradient of the loss w.r.t. the parameters of the current layer
                dtheta = l.grad(inputs[i], dL)
                # Save the gradient in the dictionary
                res[l.name] = dtheta
            # Compute the gradient of the loss w.r.t. the inputs of the current layer
            dL = l.delta(outputs[i], dL)

        return res
    


