import numpy as np
import pandas as pd
class LogisticRegressionModel:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.num_iterations):
            # Compute linear combination
            z = np.dot(X, self.weights) + self.bias
            
            # Calculate sigmoid
            y_pred = self.sigmoid(z)
            
            # Compute gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        # Compute linear combination
        z = np.dot(X, self.weights) + self.bias
        
        # Calculate sigmoid
        y_pred = self.sigmoid(z)
        
        # Convert probabilities to binary outcomes (0 or 1) using threshold 0.5
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        return y_pred_binary