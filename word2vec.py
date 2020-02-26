import pickle
from get_vocab import Vocabulary


vocab_path = './data/vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

print(vocab)
