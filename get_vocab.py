import nltk
#nltk.download('punkt')
import json as js
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
from gensim.models import Word2Vec
import os

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold, subset_id, embedding_size):
    coco = COCO(json)
    counter = Counter()
    sentences = []
    with open("data/annotations/ids_train.json", 'rb') as f:
        dic = js.load(f)
        subset_ids = dic['ids_train'] + dic['ids_val']

    # print("ids: ", len(ids))
    # print("subset ids: ", len(subset_ids))
    for i, id in enumerate(subset_ids):
        # print("id: ", id)
        caption = str(coco.anns[id]['caption'])
        sentences.append(caption.lower().split())
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # create word2vec model
    w2v = Word2Vec(sentences, min_count=1, size=embedding_size, window=5)

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab, w2v


def main(args):
    vocab, w2v = build_vocab(json=args.caption_path, threshold=args.threshold,
                             subset_id=args.subset_id_path, embedding_size=args.embedding_size)

    w2v_path = os.path.join(args.w2v_path, 'w2v_' + str(args.embedding_size) + '.model')
    vocab_path = args.vocab_path
    w2v.save(w2v_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
    print("Saved the word2vec vocab model to '{}'".format(w2v_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json', help='path for train annotation file')
    parser.add_argument('--subset_id_path', type=str,
                        default='data/annotations/ids_train.json.json', help='subset ids')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--w2v_path', type=str, default='./data/',
                        help='path for saving word2vec model')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
