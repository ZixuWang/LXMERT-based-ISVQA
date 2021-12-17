#After the download of original data file, run this script to generate new data file

import json
import numpy as np
from collections import Counter

train_orinigal_path = './imdb_nuscenes_trainval.json'  # change it to your path
train_new_path = 'trainval_with_score_quesid.json'

# modify trainingset annotation file
train_data = json.load(open(train_orinigal_path))['data']
for i, datum in enumerate(train_data):
    datum['quesid']=i
    answers = datum['answers']
    label = dict(Counter(answers))
    for answer in label.keys():
        label[answer] = np.minimum(label[answer]/2, 1)
    datum['label'] = label

with open(train_new_path, 'w') as f:
    json.dump(train_data, f)

# modify testset annotation file
test_original_path = './imdb_nuscenes_test.json'  # change it to your path
test_new_path = 'test_with_score_quesid.json'

test_data = json.load(open(test_original_path))['data']
for i, datum in enumerate(test_data):
    datum['quesid']=i
    answers = datum['answers']
    label = dict(Counter(answers))
    for answer in label.keys():
        label[answer] = np.minimum(label[answer]/2, 1)
    datum['label'] = label

with open(test_new_path, 'w') as ff:
    json.dump(test_data, ff)