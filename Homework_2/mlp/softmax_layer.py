import numpy as np

class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        """
        maximums = np.max(X,axis=1)
        p = np.exp(X - maximums[:,np.newaxis]) / np.sum(np.exp(X- maximums[:,np.newaxis]),axis=1)[:,np.newaxis]
        
        assert p.shape[0] == X.shape[0]
        assert p.shape[1] == X.shape[1]

        return p   
        """  
        maxX = np.max(X, axis=1)
        temp = np.exp(X - maxX[:,np.newaxis])
        res = temp / np.sum(temp, axis=1)[:,np.newaxis]
        
        return res
    
    
    def delta(self, Y, delta_next):
        n = Y.shape[0]
        m = Y.shape[1]
        
        y_diags = np.concatenate([np.diag(Y[r,:]).reshape(1,m,m) for r in range(n)], axis=0)
        Y_outer = np.zeros((n,m,m))
        for r in range(n):
            Y_outer[r,:,:] = np.multiply(Y[r,:][:,None], (Y[r,:].T))
        dz = y_diags - Y_outer
    
        delta = np.zeros((n,m))
        for r in range(n):
            delta[r,:] = np.dot(delta_next[r,:],dz[r,:,:])
        
        return delta
        
        
