import json

train_annotation_file = '/root/Documents/ISVQA/trainval_score_quesid.json'
test_annotation_file = '/root/Documents/ISVQA/test_score_quesid.json'

data = json.load(open(test_annotation_file))

with open(r'/root/Documents/ISVQA-Dataset/nuscenes/answers_nuscenes_more_than_1.txt', 'r') as f:
    lines = f.read().splitlines()

labellist = list(lines)
notintxt = []


for set in data:
    label = set['label']
    for i in label.keys():
        if i not in labellist:
            labellist.append(i)

# print(len(labellist))  #864
# print(labellist)
# print(len(notintxt))  # 655
with open('ans2label.txt', 'w') as ff:
    for n in labellist:
        ff.write(n + '\n')
