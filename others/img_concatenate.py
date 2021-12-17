import json
import os
from torchvision.io import read_image
import cv2


# List =  data[idx]["ImagepathList"]
# im1_front_l = cv2.imread('/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/CAM_FRONT_LEFT/n008-2018-08-30-15-52-26-0400__CAM_FRONT_LEFT__1535659467004799.jpg')
# im1_front = cv2.imread('/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/CAM_FRONT/n008-2018-08-30-15-52-26-0400__CAM_FRONT__1535659467012404.jpg')
# im_front_r =cv2.imread('/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/CAM_FRONT_RIGHT/n008-2018-08-30-15-52-26-0400__CAM_FRONT_RIGHT__1535659467020486.jpg')
# im_back_l =cv2.imread('/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/CAM_BACK_LEFT/n008-2018-08-30-15-52-26-0400__CAM_BACK_LEFT__1535659467047405.jpg')
# im_back = cv2.imread('/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/CAM_BACK/n008-2018-08-30-15-52-26-0400__CAM_BACK__1535659467037558.jpg')
# im1_back_r = cv2.imread('/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/CAM_BACK_RIGHT/n008-2018-08-30-15-52-26-0400__CAM_BACK_RIGHT__1535659467028113.jpg')

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

# im1_front_l = cv2.resize(im1_front_l, dsize=(0, 0), fx=0.5, fy=0.5)
# im1_front = cv2.resize(im1_front, dsize=(0, 0), fx=0.5, fy=0.5)
# im_front_r =cv2.resize(im1_front_r, dsize=(0, 0), fx=0.5, fy=0.5)
# im_back_l =cv2.resize(im_back_l, dsize=(0, 0), fx=0.5, fy=0.5)
# im_bcck =cv2.resize(im_bcck, dsize=(0, 0), fx=0.5, fy=0.5)
# im1_back_r =cv2.resize(im1_back_r, dsize=(0, 0), fx=0.5, fy=0.5)

def save_imgae(index,im_tile):
    path_name = '/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/test/' + str(index) + '.jpg'
    cv2.imwrite(path_name, im_tile)
    return path_name


with open("/Users/wangzixu/TUM/Forschung/ISVQA/ISVQA/input/ISVQA/imdb_nuscenes_trainval.json", 'r', encoding='UTF-8') as f:
    data = json.load(f)['data']

img_dir = '/Users/wangzixu/TUM/Forschung/ISVQA/v1.0-mini/samples/'

for idx in range(len(data)):
    image_names = data[idx]['image_names']
    image_set = []
    #     print(image_names)
    #     print(img_dir)
    for i in range(len(image_names)):  # （0，5）
        img_path = os.path.join(img_dir, image_names[i] + '.jpg')
        #         print(img_path)
        #         print(os.path.exists(img_path))
        if os.path.exists(img_path):
            image = read_image(img_path)
            image_set.append(img_path)
            if i == 5:
                print(idx)
                print(image_set)

    if (len(image_set) == 6):
        new_image = concat_tile([[cv2.imread(image_set[0]), cv2.imread(image_set[1]), cv2.imread(image_set[2])],
                                 [cv2.imread(image_set[3]), cv2.imread(image_set[4]), cv2.imread(image_set[5])]])
        new_image_path = save_imgae(idx, new_image)
        print(new_image_path)

#     data[idx]["ImagepathList"] = imge_set