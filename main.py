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
            
            # Calculate the variance matrix of the data and store it in classes_var
            # each row an example and each column a feature
            self.classes_var[str(c)] = np.cov(X_c, rowvar=False)
            
            # Calculate the prior probability of the class and store it in classes_prior
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]
        
    def predict(self, X):
        probs = np.zeros((self.number_of_examples, self.number_of_classes)) #rows of examples, where each example has a probability of belonging to each class
        #print("shape of probs", probs.shape)
        for c in range(self.number_of_classes):
            probs[:, c] = self.density_function( X, self.classes_mean[str(c)], self.classes_var[str(c)] ) + np.log(self.classes_prior[str(c)])
        return np.argmax(probs, axis=1) #returns the class with the highest log probability for each example point.
        
    def density_function(self, X, mu, sigma):
        #calculate the log of prob from multivariable gaussian distribution
        diff = np.subtract(X, mu)
        sigma_det = np.linalg.det(sigma + self.eps)
        sigma_inv = np.linalg.inv(sigma + self.eps)
        return -0.5 * self.number_of_features * np.log(2 * np.pi) - 0.5 * np.log(sigma_det) - 0.5 * np.sum(np.dot(diff, sigma_inv) * diff, axis=1)
    
if __name__ == "__main__":
    X = np.loadtxt("exampledata/data.txt", delimiter=",")
    y = np.loadtxt("exampledata/targets.txt") - 1
    
    print(X.shape)
    print(y.shape)

    NB = NaiveBayes(X, y)
    NB.fit(X, y)
    y_pred = NB.predict(X)

    print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")