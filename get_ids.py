import csv
import pickle

# with open('TrainImageIds.csv', 'r') as f:
#     reader = csv.reader(f)
#     trainIds = list(reader)
# trainIds = [int(i) for i in trainIds[0]]

# with open('TestImageIds.csv', 'r') as f:
#     reader = csv.reader(f)
#     testIds = list(reader)
# testIds = [int(i) for i in testIds[0]]

# ids = {'train_ids':trainIds, 'test_ids':testIds}

# with open("ids.pkl", 'wb') as f:
#     pickle.dump(ids, f)
with open("data/ids.pkl", 'rb') as f:
    ids = pickle.load(f)

print("ids; ", len(ids['train_ids']))


509365