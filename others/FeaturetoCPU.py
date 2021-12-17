import numpy as np
import os
import torch

featpath = []
IMGFEAT_ROOT = '/root/dataset/NuScenes/feature_output_train/'
NEW_ROOT = '/root/dataset/NuScenes/feature_train/'

for i in os.listdir(IMGFEAT_ROOT):
    if i[-8:-4] != 'info':
        featpath.append(i)

for j in featpath:
    feature = np.load(IMGFEAT_ROOT + j).item()
    feat = {}
    img_feature = []
    feat['image_id'] = feature['image_id']

    for item in feature['feature']:
        img_feature.append(item.cpu())

    feat['feature'] = img_feature
    
    np.save(os.path.join(NEW_ROOT,j), feat)

