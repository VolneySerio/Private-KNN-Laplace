import numpy as np

def laplace_mechanism(sensitivity, epsilon):
    return np.random.laplace(loc=0, scale=sensitivity/ epsilon)