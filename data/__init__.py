import numpy as np

Xgen = lambda N, M : np.random.uniform(-5,5,(N,M))

def pringle(N, M, std):
    X = Xgen(N,M)
    Y = (X[:,0]**2 - X[:,1]**2 + np.random.normal(0,std,(N,)))[...,None]    
    return X, Y

def wave(N, M, std):
    X = Xgen(N,M)
    Y = (np.cos(X[:,0]) + np.sin(X[:,1]) + np.random.normal(0,std,(N,)))[...,None]    
    return X, Y