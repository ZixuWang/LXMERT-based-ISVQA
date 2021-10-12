import json
import numpy as np
from collections import Counter

file = json.load(open('./imdb_nuscenes_trainval.json'))
data = file['data']
for i, datum in enumerate(data):
    print(f'\r{i+1}/{len(data)}', end='')
    answers = datum['answers']
    # label = {}
    # for answer in answers:
    #     label[answer] = label.get(answer, 0) + 1
    label = dict(Counter(answers))
    for answer in label.keys():
        label[answer] = np.minimum(label[answer]/2, 1)
    datum['label'] = label

# with open('./imdb_nuscenes_trainval_score.json', 'w') as f:
#     json.dump(file, f)

qus_list = []
for j, datum in enumerate(data):
    qus_id = datum['question_id']
    qus_list.append(qus_id)

uniq = np.unique(qus_list)
# if len(uniq)==len(data):
#     print('yes')
# else:
#     print('哭哭')
print('\n', len(uniq))