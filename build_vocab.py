import pandas as pd
import nltk
from nltk.corpus import words
from sklearn.model_selection import train_test_split

vocab = {}
nltk.download('words')
set_words = set(words.words())
data = pd.read_csv('data/emails.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def build_vocab(current_email):
    vocab["UNK"] = 0
    idx = len(vocab)
    for word in current_email:
        word = word.lower()
        if word not in vocab and word in set_words:
            vocab[word] = idx
            idx += 1
def build():
    for i in range(train_data.shape[0]):
        current_email = train_data.iloc[i,0].split()
        print(f"Current email is {i}/{train_data.shape[0]} and the length of vocab is curr {len(vocab)}")
        build_vocab(current_email)
        
    # Write dictionary to vocabulary.txt file
    file = open("vocabulary.txt", "w")
    file.write(str(vocab))
    file.close()
    
if __name__ == "__main__":
    build()