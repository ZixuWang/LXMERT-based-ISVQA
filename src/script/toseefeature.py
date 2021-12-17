import os
import numpy as np
import torch

# IMGFEAT_ROOT = '/root/dataset/NuScenes/feature_train/'

feat = np.load('/root/dataset/NuScenes/feature_output_train_test/n015-2018-10-08-15-52-24+0800__CAM_FRONT_RIGHT__1538985573420339.npy').item()

print(feat)
# list = []

# list.append(feat['feature'])

# feat_set = list[0]

# for i in range(len(feat_set)):
#         # obj_num += self.infoid2info[i]['num_boxes']  # total num_boxes of 6 images
#         if i == 0:
#             feats = feat_set[i]
#             # boxes = img_info[i].copy()
#         else:
#             feats = torch.cat((feats, feat_set[i]), 0)

# print(feats)

