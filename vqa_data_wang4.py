# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset

from param import args

# from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.

TINY_IMG_NUM = 512  # change to 10
FAST_IMG_NUM = 5000

# The path to data and image features.

# annotation_file = '/root/Documents/ISVQA/imdb_nuscenes_trainval_score.json'
annotation_file = '/Users/wangzixu/TUM/Forschung/data/ISVQA/imdb_nuscenes_trainval_score.json'
# test_annotation_file = ''


# VQA_DATA_ROOT = 'data/vqa/'
# IMGFEAT_ROOT = '/root/Documents/ISVQA/feature_output/'
# IMGFEAT_ROOT = '/Users/wangzixu/TUM/Forschung/ISVQA/lxmert/data/mscoco_imgfeat'
IMGFEAT_ROOT = '/Users/wangzixu/TUM/Forschung/ISVQA/ISVQA/mini_feat/'


# SPLIT2NAME = {
#     'train': 'train2014',
#     'valid': 'val2014',
#     'minival': 'val2014',
#     'nominival': 'val2014',
#     'test': 'test2015',
# }


class VQADataset:

    def __init__(self, annotation_file):  # 输入.json

        self.data = json.load(open(annotation_file))['data']
        # print(self.data)

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum  # e.g 393203003:{ } 将question_id作为索引？ -> .json文件
            for datum in self.data  # data中[]中的每一个{}是每一个datum
        }

        # # Answers
        self.ans2label = {}
        # # with open('/root/Documents/ISVQA-Dataset/nuscenes/answers_nuscenes_more_than_1.txt') as f:
        with open('/Users/wangzixu/TUM/Forschung/ISVQA/lxmert/data/vqa/answers_nuscenes_more_than_1.txt') as f:
            # lines = f.readlines()
            # print(lines)
            lines = f.read().splitlines()

            for idx, line in enumerate(lines):
                # self.ans2label[line] = idx  # 给answer贴上序号
                self.ans2label[line] = idx
            # print(self.ans2label)
            # print(len(self.ans2label)) # 650

    @property  # 通过 @property 装饰器，可以直接通过方法名来访问方法，不需要在方法名后添加一对“（）”小括号。
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

        if args.tiny:
            topk = TINY_IMG_NUM  # 512
        elif args.fast:
            topk = FAST_IMG_NUM  # 5000
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

        # 1. to have the paths of .npy files from .json file
        feat_filenames = []
        # annotation_data = json.load(open(annotation_file))['data']

        for datum in dataset.data:  # data imported from json file
            feat_filenames.append(datum['feature_paths'])

        # print(feat_filenames)

        feat_file_name = []
        feat_paths = []
        feat_info_paths = []

        # paths of feature
        for feat_filename in feat_filenames:
            for npy_name in feat_filename:
                if os.path.exists(IMGFEAT_ROOT + npy_name.split("/", 1)[1]):
                    feat_paths.append(IMGFEAT_ROOT + npy_name.split("/", 1)[1])
        # paths of feature info
        for i in feat_paths:
            if os.path.exists((i.split(".", 1)[0]) + "_info.npy"):
                feat_info_paths.append((i.split(".", 1)[0]) + "_info.npy")

        # print(feat_paths)
        # print(feat_info_paths)

        # 去除重复的path
        feat_path_new = []
        for i in feat_paths:
            if i not in feat_path_new:
                feat_path_new.append(i)

        feat_info_path_new = []
        for i in feat_info_paths:
            if i not in feat_info_path_new:
                feat_info_path_new.append(i)

        # print(feat_info_path_new)
        # print(len(feat_path_new)) #624

        # 2. 读取npy文件->列表
        # 此时image的set的结构已经不重要，因为根据image_id寻找一个set
        self.img_data = []
        self.info_data = []

        # 读取数据：for feature #according to json file
        for feat_path in feat_path_new:
            self.img_data.append(np.load(feat_path, allow_pickle=True).item())
        # print(self.img_data)
        # print(len(feat_info_paths)) # 203838

        # 读取数据：for feature info
        for feat_info_path in feat_info_path_new:
            self.info_data.append(np.load(feat_info_path, allow_pickle=True).item())
        # print(feat_info_path)
        # print(self.info_data) # [{'bbox':..., 'num_prob':..., 'image_id': ...},{},{},]

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in self.img_data:  # 遍历[ ]中的{ }
            self.imgid2img[img_datum['image_id']] = [img_datum]  # 以image_id作为索引 {1831 :[{'image_id':1381,'feature':tensor([[],[],[],[]],device = 'cuda:3')}], 1867:.....}
        # print(self.imgid2img)

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:  # 遍历annotation中的数据
            if datum['image_id'] in self.imgid2img:  # 如果anno中的image_id在.npy中，则将anno中的数据存入self.data中
                self.data.append(datum)  # self.data中是提取过特征的图片的.json
        # print("Use %d data in torch dataset" % (len(self.data)))
        # print(self.data)

        # self.infoid2info = {}
        # for info_datum in self.info_data:
        #     self.infoid2info[info_datum['image_id']] = [info_datum]  # 以image_id作为索引

        # print(self.infoid2info)

        # 合并feat和info
        # for feat_key, feat_item in self.imgid2img.items():
        #     for info_key, info_item in self.infoid2info.items():
        #         if feat_key == info_key:
        #             self.imgid2img[feat_key].append(
        #                 self.infoid2info[info_key])  # imgid2img含有feature&info的所有内容 —> 看做.tsv文件

        # print(self.imgid2img)
        """
        {'answers': ['black and white', 'black and white', 'white and black', 'black and white'],
        'ocr_tokens': [], 'image_names': ['CAM_FRONT_LEFT/n015-2018-10-02-10-50-40+0800__CAM_FRONT_LEFT__1538448754504844', 'CAM_FRONT/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448754512460',
        'CAM_FRONT_RIGHT/n015-2018-10-02-10-50-40+0800__CAM_FRONT_RIGHT__1538448754520339', 'CAM_BACK_LEFT/n015-2018-10-02-10-50-40+0800__CAM_BACK_LEFT__1538448754547423', 'CAM_BACK/n015-2018-10-02-10-50-40+0800__CAM_BACK__1538448754537525', 'CAM_BACK_RIGHT/n015-2018-10-02-10-50-40+0800__CAM_BACK_RIGHT__1538448754527893'],
         'question_str': 'what color is the curb', 'question_tokens': ['what', 'color', 'is', 'the', 'curb'],
         'feature_paths': ['CAM_FRONT_LEFT/n015-2018-10-02-10-50-40+0800__CAM_FRONT_LEFT__1538448754504844.npy', 'CAM_FRONT/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448754512460.npy', 'CAM_FRONT_RIGHT/n015-2018-10-02-10-50-40+0800__CAM_FRONT_RIGHT__1538448754520339.npy', 'CAM_BACK_LEFT/n015-2018-10-02-10-50-40+0800__CAM_BACK_LEFT__1538448754547423.npy', 'CAM_BACK/n015-2018-10-02-10-50-40+0800__CAM_BACK__1538448754537525.npy', 'CAM_BACK_RIGHT/n015-2018-10-02-10-50-40+0800__CAM_BACK_RIGHT__1538448754527893.npy'],
          'image_id': 4960, 'question_id': 25960, 'label': {'black and white': 1.0, 'white and black': 0.5}
        """

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):

        """
        to choose which

        {'answers': ['five', '<unk>', 'four', 'five'],
        'ocr_tokens': [],
        'image_names': ['CAM_FRONT_LEFT/n008-2018-08-28-16-43-51-0400__CAM_FRONT_LEFT__1535489304904799', 'CAM_FRONT/n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489304912404', 'CAM_FRONT_RIGHT/n008-2018-08-28-16-43-51-0400__CAM_FRONT_RIGHT__1535489304920494', 'CAM_BACK_LEFT/n008-2018-08-28-16-43-51-0400__CAM_BACK_LEFT__1535489304947405', 'CAM_BACK/n008-2018-08-28-16-43-51-0400__CAM_BACK__1535489304937558', 'CAM_BACK_RIGHT/n008-2018-08-28-16-43-51-0400__CAM_BACK_RIGHT__1535489304928113'],
        'question_str': 'how many vehicles are seen in the provided images', 'question_tokens': ['how', 'many', 'vehicles', 'are', 'seen', 'in', 'the', 'provided', 'images'],
        'feature_paths': ['CAM_FRONT_LEFT/n008-2018-08-28-16-43-51-0400__CAM_FRONT_LEFT__1535489304904799.npy', 'CAM_FRONT/n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489304912404.npy', 'CAM_FRONT_RIGHT/n008-2018-08-28-16-43-51-0400__CAM_FRONT_RIGHT__1535489304920494.npy', 'CAM_BACK_LEFT/n008-2018-08-28-16-43-51-0400__CAM_BACK_LEFT__1535489304947405.npy', 'CAM_BACK/n008-2018-08-28-16-43-51-0400__CAM_BACK__1535489304937558.npy', 'CAM_BACK_RIGHT/n008-2018-08-28-16-43-51-0400__CAM_BACK_RIGHT__1535489304928113.npy'],
        'image_id': 10634, 'question_id': 31634, 'label': {'five': 1.0, '<unk>': 0.5, 'four': 0.5}}

        is applied
        """

        datum = self.data[item]
        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question_str']


        # Get image info
        img_feat = []
        info_feat = []

        for i in self.img_data: # imgdata[]中有所有img的信息{}
            if i['image_id'] == img_id: # 找到选定的img set的信息{}
                img_feat.append(i['feature']) # 将信息加入img_feat
        # print(img_feat)

        for i in self.info_data: # imgdata[]中有所有img的信息{}
            if i['image_id'] == img_id: # 特定img set的信息{}
                info_feat.append(i['bbox']) # 将信息加入info_feat
        # print(info_feat)

        img_info = self.imgid2img[img_id]  # 将img_id作为img_info # img_id为XXXX的img-set的信息 #问题：只有一个set中一个img的信息
        # print(self.imgid2img[img_id])


        # obj_num = 0
        # print(len(info_feat))  # 6
        # 将一个set内的六张图的feat信息拼接
        for i in range(len(img_feat)):
            # obj_num += self.infoid2info[i]['num_boxes']  # total num_boxes of 6 images
            if i == 0:
                feats = img_feat[i]
                # boxes = img_info[i].copy()
            else:
                feats = torch.cat((feats, img_feat[i]), 0)
        # print(feats.size())  # torch.Size([600, 2048])
            # assert obj_num == len(boxes) == len(feats)

        for i in range(len(info_feat)):
            if i == 0:
                boxes = info_feat[i]
            else:
                boxes = np.concatenate((boxes, info_feat[i]), 0)
        # print(len(boxes))  # 600

        # Normalize the boxes (to 0 ~ 1)
        # img_h, img_w = img_info['image_height'], img_info['image_width']

        img_h = 900
        img_w = 1600
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        #np.testing.assert_array_less(boxes, 1+1e-5)
        #np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:  # 'label': {'five': 1.0, '<unk>': 0.5, 'four': 0.5}
            label = datum['label']
            # print(label)  # {'yes': 1.0}
            target = torch.zeros(self.raw_dataset.num_answers)
            # print(target.size())  # 650

            for ans, score in label.items():  # .items() -> ans:key score: value
                target[self.raw_dataset.ans2label[ans]] = score
                # print(self.raw_dataset.ans2label[score])
            return feats, ques, target, torch.from_numpy(boxes), ques_id
        else:
            return feats, ques, torch.from_numpy(boxes), ques_id


data1 = VQADataset(annotation_file)
data_obj36 = VQATorchDataset(dataset=data1)
data_obj36.__getitem__(1)


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):  # 我们给出ans和对应的quesid -> {"question_id": int, "answer": str }
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]  # 393203003:{ 'label': ...., question_id: ...., answers:....,}
            label = datum['label']
            if ans in label:  # 回答正确
                score += label[ans]
        return score / len(quesid2ans)  # 计算平均分

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