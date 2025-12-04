import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((b - a) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(b - a))

def minkowski_distance(a, b, p):
    return np.sum(((b - a) ** p) ** (1 / p))
