import numpy as np
import matplotlib.pyplot as plt
from collections import Counter



def euclidean_distance(a, b):
    return np.sqrt(np.sum((b - a) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(b - a))

def minkowski_distance(a, b, p):
    return np.sum(((b - a) ** p) ** (1 / p))


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, new_points):
        predictions = [self.predict_class(new_point) for new_point in new_points]
        return np.array(predictions)
    
    def predict_class(self, new_point):
        distances = [euclidean_distance(point, new_point) for point in self.X_train]

        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)[0][0]

        return most_common
    
    

