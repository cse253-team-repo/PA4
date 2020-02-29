import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import json as js
from data_loader import get_loader
from get_vocab import Vocabulary
from baseline import EncoderCNN, DecoderLSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_valid_loss(encoder, decoder, valid_loader):
    criterion = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(valid_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

    return np.mean(losses)


def main(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.ids_path, 'rb') as f:
        train_ids = js.load(f)['ids']

    with open(args.valid_ids_path, 'rb') as f:
        valid_ids = js.load(f)['ids']

    data_loader = get_loader(args.image_dir, args.caption_path, train_ids, vocab,
                             transform, args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader = get_loader(args.valid_image_dir, args.valid_caption_path, valid_ids,
                              vocab, transform, args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = EncoderCNN(args.embedding_size).to(device)
    decoder = DecoderLSTM(args.embedding_size, args.hidden_size,
                          len(vocab), args.num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(encoder.parameters()) + \
        list(encoder.bn.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    total_step = len(data_loader)
    training_losses = []
    valid_losses = []
    for epoch in range(args.num_epochs):
        training_losses_epoch = []

        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses_epoch.append(loss.item())

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                break

        training_loss = np.mean(training_losses_epoch)
        training_losses.append(training_loss)
        valid_loss = compute_valid_loss(encoder, decoder, valid_loader)
        valid_losses.append(valid_loss)
        print('Epoch {}: Training Loss = {:.4f}, Validation Loss = {:.4f}'.format(
            epoch, training_loss, valid_loss))

        # Save the best model
        if valid_loss <= np.min(valid_losses):
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-baseline.ckpt'.format(epoch+1, i+1)))
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-baseline.ckpt'.format(epoch+1, i+1)))
            print('Models Saved.')

    # Save losses as pickle
    with open(args.model_path + 'training_losses.txt', 'w') as f1:
        pickle.dump(training_losses, f1)
    with open(args.model_path + 'valid_losses.txt', 'w') as f2:
        pickle.dump(valid_losses, f2)
    print('Loss Values Saved, Training Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--ids_path', type=str,
                        default='data/annotations/ids_train.json', help='path for ids')
    parser.add_argument('--vocab_path', type=str,
                        default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='data/images/train', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--valid_ids_path', type=str,
                        default='data/annotations/ids_val.json', help='path for validation ids')
    parser.add_argument('--valid_image_dir', type=str,
                        default='data/images/test', help='directory for validation images')
    parser.add_argument('--valid_caption_path', type=str, default='data/annotations/captions_val2014.json',
                        help='path for validation annotation json file')
    parser.add_argument('--log_step', type=int, default=100,
                        help='step size for prining log info')
    # Model parameters
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
