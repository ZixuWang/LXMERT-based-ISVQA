# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
from numpy.lib.shape_base import split
import torch
from torch import tensor
from torch.utils.data import Dataset

from param import args


# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.

# The path to data and image features.


# SPLIT2NAME ={
# processed annotation file for training: path of extracted feature for training,
# processed annotation file for testing: path of extracted feature for testing,
# }

SPLIT2NAME = {
    'input/ProcessedFile/trainval_with_score_quesid.json': 'input/ISVQA/NuScenes/extracted_features/train/',
    'input/ProcessedFile/test_with_score_quesid.json': 'input/ISVQA/NuScenes/extracted_features/test/',
} 


class VQADataset:

    def __init__(self, splits:str):

        self.data = json.load(open(splits))
        self.splits = splits
        print("Load %d data from json file %s." % (len(self.data), self.splits))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['quesid']: datum
            for datum in self.data
        }

        # # Answers
        self.ans2label = {}
        self.label2ans = []
        with open('input/ProcessedFile/ans2label.txt') as f:
            lines = f.read().splitlines()
            for idx, line in enumerate(lines):
                self.ans2label[line] = idx
                self.label2ans.append(line)

    @property  
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


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

        # 1. to have the paths of .npy files from .json file
        feat_filenames = []

        for datum in dataset.data:  # data imported from json file
            feat_filenames.append(datum['feature_paths'])

        feat_paths = []
        # paths of feature
        for feat_filename in feat_filenames:
            for npy_name in feat_filename:
                if os.path.exists('%s' % (SPLIT2NAME[dataset.splits]) + npy_name.split("/", 1)[1]):
                    feat_paths.append('%s' % (SPLIT2NAME[dataset.splits]) + npy_name.split("/", 1)[1])  #203838
        
        print('load feature path in %s finished' % (SPLIT2NAME[dataset.splits]))
       
        # remove repeated paths
        feat_path_new = []
        
        for i in feat_paths:
            if i not in feat_path_new:
                feat_path_new.append(i)  # 58584

        self.img_data = []
        self.info_data = []
        self.img_feature = []
        
        for feat_path in feat_path_new:
            self.img_data.append(np.load(feat_path, allow_pickle=True).item())
            if len(self.img_data) == 300:
                print('loading %s feature finished' %len(self.img_data))
                break

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in self.img_data:
            self.imgid2img[img_datum['image_id']] = [img_datum]

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data: 
            if datum['image_id'] in self.imgid2img:  
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):

        """
        to choose which image in .json file is applied
        """

        datum = self.data[item]
        img_id = datum['image_id']
        ques_id = datum['quesid']
        ques = datum['question_str']

        # Get image info
        img_feat = []
        info_feat = []

        for i in self.img_data:
            if i['image_id'] == img_id:
                img_feat.append(i['feature'])
                info_feat.append(i['bbox'])


        img_info = self.imgid2img[img_id]

        # concatenate features of one image set
        
        for i in range(6):
            if i == 0:
                feats = img_feat[i]
            else:
                feats = torch.cat((feats, img_feat[i]), 0)  # torch.Size([600, 2048])
        
        feats = np.array(feats)

        for i in range(6):
            if i == 0:
                boxes = info_feat[i]
            else:
                boxes = np.concatenate((boxes, info_feat[i]), 0)

        img_h = 900
        img_w = 1600
        boxes = boxes.copy()
        boxes[np.where(boxes[:,0] > 1600), 0] = 1600
        boxes[np.where(boxes[:,2] > 1600), 2] = 1600
        boxes[np.where(boxes[:,1] > 900), 1] = 900
        boxes[np.where(boxes[:,3] > 900), 3] = 900

        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:  # 'label': {'five': 1.0, '<unk>': 0.5, 'four': 0.5}
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)

            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score

            return ques_id, feats, torch.from_numpy(boxes), ques, target
        else:
            return ques_id, feats, torch.from_numpy(boxes), ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict): 
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label: 
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
            for quesid, ans in quesid2ans.items():
                result.append({
                    'question_id': quesid,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)