import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X_train, y_train):
        # Convert to NumPy arrays if necessary
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        # Calculate class probabilities
        unique_classes = np.unique(y_train)
        self.class_probabilities = {cls: np.mean(y_train == cls) for cls in unique_classes}

        # Calculate feature probabilities for each class
        self.feature_probabilities = {cls: {} for cls in unique_classes}
        for cls in unique_classes:
            X_cls = X_train[y_train == cls]
            # Iterate through each feature (column)
            for feature_index in range(X_train.shape[1]):
                # Calculate mean and standard deviation for each feature
                mean = np.mean(X_cls[:, feature_index])
                std = np.std(X_cls[:, feature_index])
                # Store mean and std for this feature
                self.feature_probabilities[cls][feature_index] = (mean, std)

    def predict(self, X_test):
        # Convert to NumPy array if necessary
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()

        predictions = []
        for row in X_test:
            max_prob = float('-inf')
            predicted_class = None
            # Iterate through each class
            for cls in self.class_probabilities:
                # Calculate the log prior probability for the class
                class_prob = np.log(self.class_probabilities[cls])
                # Calculate the likelihood for each feature
                for feature_index in range(X_test.shape[1]):
                    mean, std = self.feature_probabilities[cls][feature_index]
                    # Calculate Gaussian probability for each feature
                    likelihood = self.gaussian_probability(row[feature_index], mean, std)
                    # Only add log-likelihood if likelihood > 0
                    if likelihood > 0:
                        class_prob += np.log(likelihood)
                    else:
                        # Assign a small probability value if likelihood is zero
                        class_prob += np.log(1e-9)
                # Update max_prob and predicted_class if necessary
                if class_prob > max_prob:
                    max_prob = class_prob
                    predicted_class = cls
            predictions.append(predicted_class)
        return np.array(predictions)

    def gaussian_probability(self, x, mean, std):
        if std == 0:
            # Handle case where std is zero
            return 1.0 if x == mean else 1e-9  # Return a small probability value if x == mean
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
