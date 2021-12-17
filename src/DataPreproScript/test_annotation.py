import json

train_file = json.load(open('/root/Documents/ISVQA/trainval_score_quesid.json'))
test_file = json.load(open('/root/Documents/ISVQA/test_with_score_quesid.json'))

print(train_file[1])

print('')

print(test_file[2])