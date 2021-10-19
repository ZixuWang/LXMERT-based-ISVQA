# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512 #change to 10
FAST_IMG_NUM = 5000

# The path to data and image features.

annotation_file = '/root/Documents/ISVQA/imdb_nuscenes_trainval_score.json'

# VQA_DATA_ROOT = 'data/vqa/'
IMGFEAT_ROOT = '/root/Documents/ISVQA/feature_output/feature_output/'

# SPLIT2NAME = {
#     'train': 'train2014',
#     'valid': 'val2014',
#     'minival': 'val2014',
#     'nominival': 'val2014',
#     'test': 'test2015',
# }


class VQADataset:

    def __init__(self, annotation_file): #输入.json

        self.data = json.load(open(annotation_file))['data']
        # print(self.data)

        # Convert list to dict (for evaluation)

        # Answers
        self.ans2label = {}
        with open('/root/Documents/ISVQA-Dataset/nuscenes/answers_nuscenes_more_than_1.txt') as f:
            lines = f.readlines()
            # print(lines)
            for idx, line in enumerate(lines):
                self.ans2label[idx] = line
            # print(self.ans2label)

    @property # 通过 @property 装饰器，可以直接通过方法名来访问方法，不需要在方法名后添加一对“（）”小括号。
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

data1 = VQADataset(annotation_file)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""

class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM #512
        elif args.fast:
            topk = FAST_IMG_NUM #5000
        else:
            topk = None

        # Loading detection features to img_data
        # img_data = []
        # for split in dataset.splits:
        #     # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
        #     # It is saved as the top 5K features in val2014_***.tsv
        #     load_topk = 5000 if (split == 'minival' and topk is None) else topk
        #     img_data.extend(load_obj_tsv(
        #         os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])), #选择tsv文件
        #         topk=load_topk)) #将含有前k个img特征的tsv文件传给img_data

        #1. 遍历npy文件
        feat_filenames = []
        # annotation_data = json.load(open(annotation_file))['data']

        for datum in dataset.data:
            feat_filenames.append(datum['feature_paths'])
        
        print(feat_filenames)
        feat_file_name = []
        feat_paths = []
        feat_info_paths = []

        #paths of feature
        for feat_filename in feat_filenames:
            for npy_name in feat_filename:
                feat_paths.append(IMGFEAT_ROOT + npy_name.split("/", 1)[1])
        # paths of feature info
        for i in feat_paths:
            feat_info_paths.append((i.split(".", 1)[0]) + "_info.npy")

        print(feat_info_paths)

        feat_npy = []
        for i in range(0, len(feat_file_name), 6):
            feat_npy.append(feat_file_name[i:i + 6]) #feat_npy [[],[],[],[]...]

        # 2. 读取npy文件->列表
        img_data = []
        info_data = []

        # 读取数据：for feature #according to json file
        for feat_path in feat_paths:
            img_data.append(np.load(feat_path, allow_pickle=True).item())

        # 读取数据：for feature info
        for feat_info_path in  feat_info_paths:
            info_data.append(np.load(feat_info_path, allow_pickle=True).item())

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data: #遍历[ ]中的{ }
            self.imgid2img[img_datum['image_id']] = [img_datum] # 以image_id作为索引 {image_id :["feature"...]}
            # print(self.imgid2img)

        self.infoid2info = {}
        for info_datum in info_data:
            self.infoid2info[info_datum['image_id']] = [info_datum]

        #合并feat和info
        for feat_key, feat_item in imgid2img.items():
            for info_key, info_item in infoid2info.items():
                if feat_key == info_key:
                    imgid2img[feat_key].append(infoid2info[info_key])

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data: # 遍历annotation中的数据
            if datum['image_id'] in self.imgid2img: #如果anno中的image_id在.npy中，则将anno中的数据存入self.data中
                self.data.append(datum) #重新整理dataset到self.data
        # print("Use %d data in torch dataset" % (len(self.data)))
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question_str']

        # Get image info
        img_info = self.imgid2img[img_id] # 将img_id作为img_info
        obj_num = 0
        #将六张图的feat信息拼接
        for i in range(len(img_info)):
            obj_num += img_data_info[i]['num_boxes']
            if i == 0:
            # obj_num = img_info[i]['num_boxes']
            feats = img_info[i]['feature'].copy()
            boxes = img_info[i]['bbox'].copy()
            else
            feats = tensor.cat((feats, img_info[i]['feature'].copy()), 1)
            boxes = np.concatenate((boxes, img_info[i]['bbox'].copy()), 1)
            assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['image_height'], img_info['image_width']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items(): # .items() ans:key score:value
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target, torch.from_numpy(boxes)
        else:
            return ques_id, feats, boxes, ques, torch.from_numpy(boxes)

data_obj36 = VQATorchDataset(dataset = data1)

class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):#我们给出ans和对应的quesid -> {"question_id": int, "answer": str }
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid] #393203003:{ }
            label = datum['label']
            if ans in label: #回答正确
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


