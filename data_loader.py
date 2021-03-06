import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from get_vocab import Vocabulary
from pycocotools.coco import COCO
import json as js
import pdb

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, ids, vocab, transform):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = ids
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        # print("caption: ", caption)
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            # check the size of the image. If less than the threshold, resize it.
            if min(image.size) < self.transform.transforms[0].size[0]:
                new_size = (256, 256)
                image = image.resize(new_size, Image.ANTIALIAS)
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, ann_id

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ann_id = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, ann_id


def get_loader(root, json, ids, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       ids=ids,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == "__main__":
    root_train = "data/images/train"
    json_train = "data/annotations/captions_train2014.json"

    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print(vocab.word2idx)
    print(vocab.idx2word)
    
    with open("data/annotations/ids_train.json", 'rb') as f:
        # print(js.load(f).keys())
        dic = js.load(f)

        ids_train = dic['ids_train']
        ids_val = dic['ids_val']

        print('ids train: ', len(ids_train))
        print('ids val: ', len(ids_val))

    '''
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    train_loader = get_loader(root_train, json_train,
                              ids_train, vocab, transform, 4, False, 1)
    val_loader = get_loader(root_train, json_train,
                              ids_val, vocab, transform, 4, False, 1)

    for i, (images, captions, lengths) in enumerate(train_loader):
        pdb.set_trace()
        print("images shape:", images.shape)
        print("captions: ", captions)
        print("lengths: ", lengths)
        break

    for i, (images, captions, lengths) in enumerate(val_loader):
        print("images shape:", images.shape)
        print("captions: ", captions)
        print("lengths: ", lengths)
        break
    '''
