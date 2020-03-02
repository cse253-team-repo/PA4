import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import json as js
from data_loader import get_loader
from get_vocab import Vocabulary
from baseline import EncoderCNN, DecoderLSTM, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
# from pycocotools.coco import COCO
import json as js
import csv
import tqdm
from pycocotools.coco import COCO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gen_captions(captions, sampled_ids, vocab):
    batch_size = sampled_ids.shape[0]
    gt = [''] * batch_size
    preds = [''] * batch_size
    for sample_id in range(batch_size):
        for gt_token_id in captions[sample_id]:
            if vocab.idx2word[gt_token_id.item()] == '<start>':
                continue
            elif vocab.idx2word[gt_token_id.item()] == '<end>':
                break
            else:
                gt[sample_id] += vocab.idx2word[gt_token_id.item()]
                gt[sample_id] += ' '
        gt[sample_id] = gt[sample_id][:-3] +'.'

        for pred_token_id in sampled_ids[sample_id]:
            if vocab.idx2word[pred_token_id.item()] == '<start>':
                continue
            elif vocab.idx2word[pred_token_id.item()] == '<end>':
                break
            else:
                preds[sample_id] += vocab.idx2word[pred_token_id.item()]
                preds[sample_id] += ' '
        preds[sample_id] = preds[sample_id][:-3] + '.'
    return gt, preds

class evaluate_captions:
    def __init__(self):
        with open('models/evaluation_results.json', 'r') as f:
            self.results = js.load(f)
    
        with open('TestImageIds.csv', 'r') as f_image_id:
            self.reader = csv.reader(f_image_id)
            self.testIds = list(reader)
        
        with open("data/annotations/ids_val.json", 'rb') as f:
            self.dic = js.load(f)
            self.ids_val = dic['ids']
            
        self.testIds = [int(i) for i in testIds[0]]
        self.coco = COCO("data/annotations/captions_val2014.json")

        self.candidates = list(results.keys())
    def step(self):
        score1 = 0
        score4 = 0

        smoother = SmoothingFunction()

        num = 0 
        for i in self.candidates:
            print(int(i))
            img_id = self.coco.anns[int(i)]['image_id']
            references = []
            for entry in self.coco.imgToAnns[img_id]:
                references.append(entry['caption'])
            score1 += sentence_bleu(references, results[i], weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
            score4 += sentence_bleu(references, results[i], weights=(0, 0, 0, 1), smoothing_function=smoother.method1)
            num += 1
        bleu1 = 100*score1/num
        bleu4 = 100*score4/num
        
        return bleu1, bleu4



class test:
    def __init__(self, args):
        if not os.path.exists(args.model_path):
            os.mkdir(args.model_path)
        self.transform = transforms.Compose([
                            transforms.RandomCrop(args.crop_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))]
                            )
        with open(args.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(args.ids_path, 'rb') as f:
            dic = js.load(f)
            self.train_ids = dic['ids_train']
            self.val_ids = dic['ids_val']
        with open(args.valid_ids_path, 'rb') as f:
            self.test_ids = js.load(f)['ids']
        self.train_loader = get_loader(args.image_dir, args.caption_path, self.train_ids, self.vocab,
                                 self.transform, args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.test_loader = get_loader(args.image_dir, args.caption_path, self.train_ids, self.vocab,
                                 self.transform, args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluate_cap = evaluate_captions()

    def step(self):
        model = CaptionCNNRNN(self.args, len(self.vocab)).to(self.device)
        model.load_model_state(self.args.model_path)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_step = len(self.test_loader)
        test_losses = []
        perplexities = []
        captions_test = {}

        for i, (images, captions, lengths, ann_id) in enumerate(tqdm(self.test_loader)):
            # print("!!!!!!!!!!!!", ann_id)
            model.eval()

            images = images.to(self.device)
            captions = captions.to(self.device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            sampled_ids = decoder.sample(features)
            test_losses.append(loss.item())
            perplexities.append(np.exp(loss.item()))
            gt, preds = gen_captions(captions, sampled_ids, self.vocab)
            counter = 0
            for j in ann_id:
                captions_test[j] = preds[counter]
                counter += 1
        test_loss = np.mean(test_losses)
        perplexity = np.mean(perplexities)
        print("test loss: ", test_loss)
        print("perple: ", perplexity)
        with open(self.args.model_path + 'evaluation_results.json', 'w') as f1:
            js.dump(captions_test, f1)
        BLEU1, BLEU4 = self.evaluate_cap.step()
        print("BLEU 1:", np.round(bleu1,2), 
              "BLEU 4:", np.round(bleu4,2))
        print('Testing Finished.')




if __name__ == '__main__':
    args = load_config(path="config/test.yaml")
    args = AttrDict(args)
    main(args)
