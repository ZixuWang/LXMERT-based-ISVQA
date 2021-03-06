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

from others.param2 import args

# CUDA_VISIBLE_DEVICES= 0
# from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.

TINY_IMG_NUM = 10  # change to 10
FAST_IMG_NUM = 5000

# The path to data and image features.

# annotation_file = '/root/Documents/ISVQA/trainval_score_quesid.json'

train = '/root/Documents/ISVQA/trainval_score_quesid.json'
# test = '/root/Documents/ISVQA/test_score_quesid.json'


# IMGFEAT_ROOT = '/root/dataset/NuScenes/feature_output_train_test/'

SPLIT2NAME = {
    '/root/Documents/ISVQA/trainval_score_quesid.json': '/root/dataset/NuScenes/feature_output_train_test/',
    '/root/Documents/ISVQA/test_score_quesid.json': '/root/dataset/NuScenes/feature_output_test_test_test/',
}

class VQADataset:

    def __init__(self, splits:str):  # 输入.json

        self.data = json.load(open(splits))
        self.splits = splits
        print("Load %d data from json file %s." % (len(self.data), self.splits))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['quesid']: datum  # 将question_id作为索引
            for datum in self.data
        }

        # # Answers
        self.ans2label = {}
        self.label2ans = []
        with open('/root/Documents/lxmert/ans2label.txt') as f:
        # with open('/root/Documents/ISVQA-Dataset/nuscenes/answers_nuscenes_more_than_1.txt') as f:
            lines = f.read().splitlines()

            for idx, line in enumerate(lines):
                # self.ans2label[line] = idx  # 给answer贴上序号
                self.ans2label[line] = idx
                self.label2ans.append(line)
            # print(self.ans2label)
            # print('')
            # print(self.label2ans)
            # print(len(self.ans2label)) # 650
            # print(len(self.label2ans))

    @property  # 直接通过方法名来访问方法
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

        # feat_file_name = []
        feat_paths = []
        # feat_info_paths = []
        # paths of feature
        for feat_filename in feat_filenames:
            for npy_name in feat_filename:
                if os.path.exists('%s' % (SPLIT2NAME[dataset.splits]) + npy_name.split("/", 1)[1]):
                    feat_paths.append('%s' % (SPLIT2NAME[dataset.splits]) + npy_name.split("/", 1)[1])  #203838
        
        print('load feature path in %s finished' % (SPLIT2NAME[dataset.splits]))
                
        # paths of feature info
        # for i in feat_paths:
        #     if os.path.exists((i.split(".", 1)[0]) + "_info.npy"):
        #         feat_info_paths.append((i.split(".", 1)[0]) + "_info.npy")  #203826

        # 去除重复的path
        feat_path_new = []
        
        for i in feat_paths:
            if i not in feat_path_new:
                feat_path_new.append(i)  #58584


        # feat_info_path_new = []
        # for i in feat_info_paths:
        #     if i not in feat_info_path_new:
        #         feat_info_path_new.append(i)  #58581

        # 2. 读取npy文件->列表
        # 此时image的set的结构已经不重要，因为根据image_id寻找一个set
        self.img_data = []
        self.info_data = []
        self.img_feature = []
        

        # 读取数据：for feature #according to json file
        # while len(self.info_data) < 5:
        # for feat_path in feat_path_new:
        #     self.img_data.append(np.load(feat_path, allow_pickle=True).item())
        #     if len(self.img_data) > 300:
        #         break


        #     # 读取数据：for feature info
        # for feat_info_path in feat_info_path_new:
        #     self.info_data.append(np.load(feat_info_path, allow_pickle=True).item())
        #     if len(self.info_data) > 50:
        #         break
        for feat_path in feat_path_new:
            self.img_data.append(np.load(feat_path, allow_pickle=True).item())
            # if len(self.img_data) == 3000:
            #     print('loading %s feature finished' %len(self.img_data))
            #     break

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in self.img_data:  # 遍历[ ]中的{ }
            self.imgid2img[img_datum['image_id']] = [img_datum]  # 以image_id作为索引 {1831 :[{'image_id':1381,'feature':tensor([[],[],[],[]]), 'bbox': }], 1867:.....}

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:  # 遍历annotation中的数据
            if datum['image_id'] in self.imgid2img:  # 如果anno中的image_id在.npy中，则将anno中的数据存入self.data中
                self.data.append(datum)  # self.data中是提取过特征的图片的.json
        print("Use %d data in torch dataset" % (len(self.data)))
        # print(self.data[1])

        # self.infoid2info = {}
        # for info_datum in self.info_data:
        #     self.infoid2info[info_datum['image_id']] = [info_datum]  # 以image_id作为索引

        # print(self.infoid2info)

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
        to choose which image in .json file is applied
        """

        datum = self.data[item]  #the item-th in .json file
        img_id = datum['image_id']
        ques_id = datum['quesid']
        ques = datum['question_str']


        # Get image info
        img_feat = []
        info_feat = []

        for i in self.img_data: # to find img的信息{'image_id':1234, 'feature':[tensor1, ..., tensor6]}
            if i['image_id'] == img_id:  # 找到选定的img set的信息{}
                img_feat.append(i['feature'])  # 将信息加入img_feat
                info_feat.append(i['bbox'])

        # for i in self.img_data: # to find img的信息{'image_id':1234, 'feature':[tensor1, ..., tensor6]}
        #     if i['image_id'] == img_id: # 找到选定的img set的信息{}
        #         img_feat.append(i['feature']) # 将信息加入img_feat
        #         break
        # img_feat = img_feat[0]
        # print(img_feat)  # [tensor1,tensor2,tensor3,tensor4,tensor5,tensor6]


        # for i in self.info_data: # imgdata[]中有所有img的信息{}
        #     if i['image_id'] == img_id: # 特定img set的信息{}
        #         info_feat.append(i['bbox']) # 将信息加入info_feat  # [array1,array2,array3,array4,array5,array6]
        # print(info_feat)

        img_info = self.imgid2img[img_id]  # 将img_id作为img_info # img_id为XXXX的img-set的信息 #问题：只有一个set中一个img的信息
        # print(self.imgid2img[img_id])

        # 将一个set内的六张图的feat信息拼接
        
        for i in range(6):
            # obj_num += self.infoid2info[i]['num_boxes']  # total num_boxes of 6 images
            if i == 0:
                feats = img_feat[i]
                # boxes = img_info[i].copy()
            else:
                feats = torch.cat((feats, img_feat[i]), 0)  # torch.Size([600, 2048])
        
        feats = np.array(feats)
        # assert obj_num == len(boxes) == len(feats)
    
    

        for i in range(6):
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
            # print(label)  # {'yes': 1.0}
            target = torch.zeros(self.raw_dataset.num_answers)
            # print(target.size())  # 650

            for ans, score in label.items():  # .items() -> ans:key score: value
                target[self.raw_dataset.ans2label[ans]] = score

            return ques_id, feats, torch.from_numpy(boxes), ques, target
        else:
            return ques_id, feats, torch.from_numpy(boxes), ques



# data1 = VQADataset(train)
# data_obj36 = VQATorchDataset(dataset=data1)
# data_obj36.__getitem__(12)


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):  # 我们给出ans和对应的quesid -> {"question_id": int, "answer": str }
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]  # 123:{ 'label': ...., question_id: ...., answers:....,}, quesid对应的datum
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
            for quesid, ans in quesid2ans.items():
                result.append({
                    'question_id': quesid,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)