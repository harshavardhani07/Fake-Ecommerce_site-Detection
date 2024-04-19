import numpy as np
import pandas as pd
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        # Convert y to a numpy array to avoid KeyError issues
        self.X_train = X
        self.y_train = np.array(y)
        
    def predict(self, X):
        # Predict the label for each data point in X
        predictions = []
        for x in X:
            # Calculate distances between x and all points in X_train
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Find the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Retrieve the labels of the k nearest neighbors
            k_labels = self.y_train[k_indices]  # Use array indexing
            # Determine the most common label (mode) among the k nearest neighbors
            most_common_label = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return np.array(predictions)
    
    def accuracy(self, y_true, y_pred):
        # Calculate the accuracy of the model
        return np.mean(y_true == y_pred)