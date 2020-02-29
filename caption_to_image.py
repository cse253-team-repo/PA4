import csv
import pickle
import json


with open('TrainImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    trainIds = list(reader)
trainIds = [int(i) for i in trainIds[0]]
num_im = len(trainIds)
trainset_id = trainIds[:int(num_im*0.8)]
valset_id = trainIds[int(num_im*0.8):]
print("trainset im: ", len(trainset_id))
print("valset im: ", len(valset_id))


with open('./data/annotations/captions_train2014.json', 'r') as f:
    annotations = json.load(f)

print("number of train annotations: ", len(annotations['annotations']))
cap_to_im_train = []
cap_to_im_val = []

for i in annotations['annotations']:
    dic ={}
    # print(i)
    cap_id = i['id']
    im_id = i['image_id']
    if im_id in trainset_id:
        # cap = i['caption']
        # dic[int(cap_id)] = im_id
        cap_to_im_train.append(cap_id)
    if im_id in valset_id:
        cap_to_im_val.append(cap_id)

ids = {}
ids['ids_train'] = cap_to_im_train
ids['ids_val'] = cap_to_im_val

with open('./data/annotations/ids_train.json', 'w') as wf:
    json.dump(ids, wf)

print("training_subsets: ", len(ids['ids_train']))
print("validation_subsets: ", len(ids['ids_val']))
"""
with open('TestImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    testIds = list(reader)
testIds = [int(i) for i in testIds[0]]

with open('./data/annotations/captions_val2014.json', 'r') as f:
    annotations = json.load(f)

print("number of test annotations: ", len(annotations['annotations']))
cap_to_im = []

for i in annotations['annotations']:
    # print(i)
    dic = {}
    cap_id = i['id']
    im_id = i['image_id']
    if im_id in testIds:
        # cap = i['caption']
        # dic[cap_id] = im_id
        cap_to_im.append(cap_id)
ids = {}
ids['ids']=cap_to_im
with open('./data/annotations/ids_val.json', 'w') as wf:
    json.dump(ids, wf)
print("subset test: ", len(ids['ids']))
"""