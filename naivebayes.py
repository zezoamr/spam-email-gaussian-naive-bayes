import numpy as np

class NaiveBayes:
    
    def __init__(self, X, y) -> None:
        # Get the number of features and examples in the input data
        self.number_of_examples, self.number_of_features = X.shape

        # Get the number of classes in the target labels
        self.number_of_classes = len(np.unique(y))

        # Set a small epsilon value for numerical stability
        self.eps = 1e-6
        
        # Initialize dictionaries to store mean, variance, and prior values for each class
        self.classes_mean = {}
        self.classes_var = {}
        self.classes_prior = {}
        
        #print("number of classes", self.number_of_classes)
        #print("number of features", self.number_of_features)
        #print("number of examples", self.number_of_examples)
    
    def fit(self, X, y):
        # Iterate over each class in X
        for c in range(self.number_of_classes):
            # Get the data for the current class
            X_c = X[c == y]
            
            # Calculate the mean of the data along axis 0 and store it in classes_mean
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            
            # Calculate the variance of the data along axis 0 and store it in classes_var
            # Use np.var instead of np.cov to get a diagonal covariance matrix
            self.classes_var[str(c)] = np.var(X_c, axis=0)
            
            # Calculate the prior probability of the class and store it in classes_prior
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]
        
    def predict(self, X):
        probs = np.zeros((self.number_of_examples, self.number_of_classes)) #rows of examples, where each example has a probability of belonging to each class
        #print("shape of probs", probs.shape)
        for c in range(self.number_of_classes):
            probs[:, c] = self.density_function( X, self.classes_mean[str(c)], self.classes_var[str(c)] ) + np.log(self.classes_prior[str(c)])
        return np.argmax(probs, axis=1) #returns the class with the highest log probability for each example point.
        
    def density_function(self, X, mu, sigma):
        # In the original version of your code, you were trying to calculate this probability using a full covariance matrix,
        # which represents the variances and covariances of each feature in your data. However, this led to an error because your covariance matrix was singular,
        # meaning it didn’t have an inverse.
        # To fix this issue, we switched to using a diagonal covariance matrix
        #calculate the log of prob from multivariable gaussian distribution
        diff = np.subtract(X, mu)
        sigma_inv = 1.0 / (sigma + self.eps)
        return -0.5 * self.number_of_features * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps)) - 0.5 * np.sum((diff ** 2) * sigma_inv, axis=1)