import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import json as js
from data_loader import get_loader
from get_vocab import Vocabulary
from baseline import *
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from attrdict import AttrDict
from utils import *
import tqdm
import pdb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_valid_loss(model, valid_loader, vocab):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(valid_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]
            outputs = model(images, captions, lengths)
            sample_ids = model.sample(features)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

    num_samples = 3
    gt = [''] * num_samples
    preds = [''] * num_samples
    for sample_id in range(num_samples):
        for gt_token_id in captions[sample_id]:
            gt[sample_id] += vocab.idx2word[gt_token_id.item()]
            gt[sample_id] += ' '

        for pred_token_id in sample_ids[sample_id]:
            preds[sample_id] += vocab.idx2word[pred_token_id.item()]
            preds[sample_id] += ' '

    print("Ground truth: {}".format(gt))
    print("Predict: {}".format(preds))
    return np.mean(losses)


class Train:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(args.model_path):
            os.mkdir(args.model_path)

        # define the transformation 
        self.transform = transforms.Compose([
                        transforms.RandomCrop(args.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))]
                        )

        self.vocab, self.train_ids, self.val_ids, self.test_ids = self.load_data_all(args)

        self.train_loader = get_loader(args.image_dir, args.caption_path, self.train_ids, self.vocab,
                                self.transform, args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        self.valid_loader = get_loader(args.image_dir, args.caption_path, self.val_ids, self.vocab,
                                 self.transform, args.batch_size, shuffle=False, num_workers=args.num_workers)

        self.test_loader = get_loader(args.valid_image_dir, args.valid_caption_path, self.test_ids,
                                  self.vocab, self.transform, args.batch_size, shuffle=False, num_workers=args.num_workers)


    def step(self):
        model = CaptionCNNRNN(self.args, len(self.vocab)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)
        training_losses = []
        valid_losses = []
        for epoch in range(self.args.num_epochs):
            training_losses_epoch = []

            for i, (images, captions, lengths) in enumerate(self.train_loader):
                model.train()
                images = images.to(self.device)
                captions = captions.to(self.device)
                targets = pack_padded_sequence(
                    captions, lengths, batch_first=True)[0]

                outputs = model(images, captions, lengths)
                loss = criterion(outputs, targets)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                training_losses_epoch.append(loss.item())
                if i % self.args.log_step == 0:
                    print("Epoch: {}, Iteration: {}, Loss: {}, Perplexity: {}".format(epoch, i, np.exp(loss.item())))
            training_loss = np.mean(training_losses_epoch)
            training_losses.append(training_loss)
            valid_loss = compute_valid_loss(model, self.valid_loader, vocab)
            valid_losses.append(valid_loss)
            print('Epoch {}: Training Loss = {:.4f}, Validation Loss = {:.4f}'.format(
                epoch, training_loss, valid_loss))

            # Save the best model
            if valid_loss <= np.min(valid_losses):
                torch.save(encoder.state_dict(), os.path.join(
                    self.args.model_path, 'encoder-baseline'.format(epoch+1, i+1)))
                torch.save(decoder.state_dict(), os.path.join(
                    self.args.model_path, 'decoder-baseline'.format(epoch+1, i+1)))
                print('Models Saved.')
            # Save losses as pickle
            with open(self.args.model_path + 'training_losses.txt', 'wb') as f1:
                pickle.dump(training_losses, f1)
            with open(self.args.model_path + 'valid_losses.txt', 'wb') as f2:
                pickle.dump(valid_losses, f2)

    def load_data_all(self, args):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(self.args.ids_path, 'rb') as f:
            dic = js.load(f)
            train_ids = dic['ids_train']
            val_ids = dic['ids_val']
        with open(self.args.valid_ids_path, 'rb') as f:
            test_ids = js.load(f)['ids']
        return vocab, train_ids, val_ids, test_ids




def main(args):
    train = Train(args)
    train.step()

if __name__ == '__main__':
    args = load_config(path="config/config.yaml")
    args = AttrDict(args)
    main(args)
