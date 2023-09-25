from naivebayes import NaiveBayes
from create_frequency_vectors import create_frequency_vectors
from build_vocab import build
import numpy as np

def main(test=False, create_vocab=False):
    if test: 
        X = np.loadtxt("exampledata/data.txt", delimiter=",")
        y = np.loadtxt("exampledata/targets.txt") - 1
    else:
        if create_vocab:
            build()
            create_frequency_vectors()
        X = np.load("data/X.npy")
        y = np.load("data/y.npy")
    
    print(X.shape)
    print(y.shape)

    NB = NaiveBayes(X, y)
    NB.fit(X, y)
    y_pred = NB.predict(X)

    print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")
    
if __name__ == "__main__":
    main()