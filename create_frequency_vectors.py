import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/emails.csv")
# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

file = open("vocabulary.txt", "r")
contents = file.read()
vocabulary = ast.literal_eval(contents)


X = np.zeros((test_data.shape[0], len(vocabulary)))
y = np.zeros((test_data.shape[0]))

def create_frequency_vectors():
    for i in range(test_data.shape[0]):
        email = test_data.iloc[i, 0].split() 

        for email_word in email:
            if email_word.lower() in vocabulary:
                X[i, vocabulary[email_word]] += 1
            else :
                X[i, vocabulary["UNK"]] += 1

        y[i] = test_data.iloc[i, 1]
        
    # Save stored numpy arrays
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)

if __name__ == "__main__":
    create_frequency_vectors()