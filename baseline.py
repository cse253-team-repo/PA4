import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import WeightedRandomSampler
from torch.autograd import Variable
from gensim.models import Word2Vec
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


class DecoderLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers, use_word2vec=False):
        super(DecoderLSTM, self).__init__()

        if use_word2vec == True:
            try:
                w2v = Word2Vec.load(
                    './data/w2v_' + str(embedding_size) + '.model')
                self.embedding = nn.Embedding.from_pretrained(w2v.wv)
                for param in self.embedding.parameters():
                    param.requires_grad = False
            except:
                raise NotImplementedError
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_length = 10  # max_length

    def forward(self, features, captions, lengths, states=None):
        hiddens = []
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        inputs_iter = packed[0]
        batch_size_iter = packed[1]

        # for i in range(max(lengths)):
        # for i in range(len(packed)):
        #     # hidden, states = self.lstm(embeddings[:,i].unsqueeze(1), states)
        #     print("hidden shape: ", hidden.shape)
        #     hiddens.append(hidden)
        # hiddens = torch.cat(hiddens,dim=1)
        # print("hiddens shape: ", hiddens.shape)
        # hiddens = Variable(hiddens, requires_grad=True)

        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None, stochastic=False):
        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(self.max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            if stochastic == True:
                predicted = WeightedRandomSampler(
                    torch.nn.functional.softmax(outputs, dim=2), outputs.shape[2])
            else:
                _, predicted = outputs.max(1)

            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers, use_word2vec=False):
        super(DecoderRNN, self).__init__()

        if use_word2vec == True:
            try:
                w2v = Word2Vec.load(
                    './data/w2v_' + str(embedding_size) + '.model')
                self.embedding = nn.Embedding.from_pretrained(w2v.wv)
                for param in self.embedding.parameters():
                    param.requires_grad = False
            except:
                raise NotImplementedError
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.rnn = nn.RNN(embedding_size, hidden_size,
                          num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_length = 10  # max_length

    def forward(self, features, captions, lengths, states=None):
        hiddens = []
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        inputs_iter = packed[0]
        batch_size_iter = packed[1]

        hiddens, _ = self.rnn(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None, stochastic=False):
        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(self.max_length):
            hiddens, states = self.rnn(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            if stochastic == True:
                predicted = WeightedRandomSampler(
                    torch.nn.functional.softmax(outputs, dim=2), outputs.shape[2])
            else:
                _, predicted = outputs.max(1)

            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
