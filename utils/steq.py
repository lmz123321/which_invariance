import numpy as np
import torch
from tensor import dfTotensor

def normal(mu,sigma,size):
    return np.random.normal(loc=mu, scale=sigma, size=size)

def uniform(low, high, size):
    return np.random.uniform(low=low, high=high, size=size)

def duniform(low, high, size):
    """
    generate random numbers from a Uniform distribution [low,high]U[-high,-low]
    """
    randsign = 2*np.random.randint(0,2,size)-1
    return randsign*np.random.uniform(low=low, high=high, size=size)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def identity(x):
    return x

class StEq():
    r"""
    Functs: - structural equation
            - nonlin is a function
            - lin is a list of parameters
    """
    def __init__(self,nonlin, lin):
        self.nonlin = nonlin
        self.lin = torch.from_numpy(np.array(lin).reshape(1,-1))
    
    def __call__(self,covariates):
        return self.nonlin(torch.sum((self.lin*covariates),dim=1)).view(-1,1).float()
    
    def predict(self, covariates):
        covariates = dfTotensor(covariates)
        return self.nonlin(torch.sum((self.lin*covariates),dim=1)).detach().numpy()
