import numpy as np
from distances import euclidean_distance

class RN:
    def __init__(self, epsilon=1.0):
        self.epsilon =  epsilon

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        self.n_classe = len(self.classes_)
        
    def predict(self, new_point, r ):
        distances = [euclidean_distance(point, new_point) for point in self.X_train]

        noisy_counts = {}
        for cls in self.classes_:  

            count = np.sum((self.y_train == cls) & (distances <= r))
            
            scale = self.n_classe / self.epsilon
        
            noise = np.random.laplace(loc=0, scale=scale)
            noisy_counts[cls] = count + noise
        
        return noisy_counts
    
    