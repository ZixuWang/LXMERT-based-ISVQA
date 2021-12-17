# This file is to generate a new answers file which contains all answers appeared in train and test dataset 


import json

train_annotation_file = 'input/ISVQA/imdb_nuscenes_trainval.json'
test_annotation_file = 'input/ISVQA/imdb_nuscenes_test.json'

data = json.load(open(test_annotation_file))

with open(r'input/ISVQA/answers_nuscenes_more_than_1.txt', 'r') as f:
    lines = f.read().splitlines()

labellist = list(lines)
notintxt = []


for set in data:
    label = set['label']
    for i in label.keys():
        if i not in labellist:
            labellist.append(i)

with open('input/ISVQA/ans2label.txt', 'w') as ff:
    for n in labellist:
        ff.write(n + '\n')
