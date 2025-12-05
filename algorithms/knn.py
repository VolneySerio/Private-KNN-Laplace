import numpy as np
from collections import Counter
from .distances import euclidean_distance
from tqdm import tqdm

class KNN:
    def __init__(self, k, verbose=True):
        self.k = k
        self.verbose = verbose

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, new_points):
        if self.verbose:

            predictions = []
            for new_point in tqdm(new_points, desc="KNN Predict", 
                                 unit="sample", leave=True):
                predictions.append(self.predict_class(new_point))
        else:
            
            predictions = [self.predict_class(new_point) for new_point in new_points]
        
        return np.array(predictions)
    
    def predict_class(self, new_point):
        distances = [euclidean_distance(point, new_point) for point in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
