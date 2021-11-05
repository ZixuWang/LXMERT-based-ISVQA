# Copyright (c) Facebook, Inc. and its affiliates.

# Requires vqa-maskrcnn-benchmark (https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark)
# to be built and installed. Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
# When the --background flag is set, the index saved with key "objects" in
# info_list will be +1 of the Visual Genome category mapping above and 0
# is the background class. When the --background flag is not set, the
# index saved with key "objects" in info list will match the Visual Genome
# category mapping.
import argparse
import os
import json

import cv2 # opencv库
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from mmf.utils.download import download
from PIL import Image

from tools.scripts.features.extraction_utils import chunks, get_image_files


class FeatureExtractor:

    MODEL_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model.pth",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model_x152.pth",
    }
    CONFIG_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model.yaml",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model_x152.yaml",
    }

    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self._try_downloading_necessities(self.args.model_name)
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def _try_downloading_necessities(self, model_name):
        if self.args.model_file is None and model_name is not None:
            model_url = self.MODEL_URL[model_name]
            config_url = self.CONFIG_URL[model_name]
            self.args.model_file = model_url.split("/")[-1]
            self.args.config_file = config_url.split("/")[-1]
            if os.path.exists(self.args.model_file) and os.path.exists(
                self.args.config_file
            ):
                print(f"model and config file exists in directory: {os.getcwd()}")
                return
            print("Downloading model and configuration")
            download(model_url, ".", self.args.model_file)
            download(config_url, ".", self.args.config_file)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name", default="X-152", type=str, help="Model to use for detection"
        )
        parser.add_argument(
            "--model_file",
            default=None,
            type=str,
            help="Detectron model file. This overrides the model_name param.",
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))
        # checkpoint 得到一个字典
        # model.load_state_dict(checkpoint.pop("model"))
        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()  # 测试模式
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:  # 如果通道数大于3，（H，W，C），-1->C
            im = np.array(img.convert("RGB")).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  
            #灰度图转RGB，(H,W)->(H,W,1)->(H,W,3)

        im = im[:, :, ::-1] #::-1 首尾颠倒，RGB->BGR
        im -= np.array([102.9801, 115.9465, 122.7717]) #减去3通道的平均值 BGR
        im_shape = im.shape  # im是array，格式是RGB；im.shape是(H,W,C)
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2]) # [0:2], [0,2)->0,1
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE: # np.round() 取整
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        # numpy->Tensor; Tensor(H,W,C)->(C,H,W) 
        img = torch.from_numpy(im).permute(2, 0, 1)
    
        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        # output[0]['proposals'], (B,N,F), B:batch_size; N:number of boxes
        # 获取输出proposals的候选框
        batch_size = len(output[0]["proposals"]) 
        # 输出每一张图片的proposals的box长度
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        # 输出置信度的得分
        score_list = output[0]["scores"].split(n_boxes_per_image)
        # 对置信度缩放到0-1
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]  # 对最后一维求softmax
        # 输出每一张图片中n个box的特征embedding
        feats = output[0][feature_name].split(n_boxes_per_image)
        
        cur_device = score_list[0].device

        feat_list = []
        info_list = []
        
        # 依次处理proposals中的候选框信息
        for i in range(batch_size):
            # 对输出的候选框除以缩放的尺度
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            # 取到score_list中每一个score的得分
            scores = score_list[i]
            # max_conf 为最大置信度
            max_conf = torch.zeros(scores.shape[0]).to(cur_device)
            # 创建置信度过滤矩阵
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            # 开始的index为1
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            # 得分矩阵scores matrix的第0列, 对应于background类
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                # 将IOU交并比小于0.5的去掉
                keep = nms(dets, cls_scores, 0.5)
                # 将比max_conf大的进行保留
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater
                    # than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            # 对类别概率分值和索引进行排序
            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            # box的数目
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            # 每一个box的索引值
            keep_boxes = sorted_indices[: self.args.num_features]
            # 特征list添加
            # TODO
            feat_list.append(feats[i][keep_boxes])
            # bbox的得分
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            # 使用得分预测类别标签
            objects = torch.argmax(scores[keep_boxes][:, start_index:], dim=1)
            # 将结果信息添加进列表中返回
            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "cls_prob": scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                }
            )

        return feat_list, info_list
    # ['img/1.png', 'img/2.png', 'img/3.png']
    # [['1','2'], ['3', '4']]
    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        for image_path in image_paths:
            im, im_scale, im_info = self._image_transform(image_path) 
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)  #Tensor转换成ImageList
        current_img_list = current_img_list.to("cuda")  # cuda: GPU

        with torch.no_grad():  # 只是测试，不需要权重更新，不需要梯度
            output = self.detection_model(current_img_list)

        feat_list, info_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list, info_list

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        np.save(
            os.path.join(self.args.output_folder, file_base_name), feature.cpu().numpy()
        )
        np.save(os.path.join(self.args.output_folder, info_file_base_name), info)

    def extract_features(self):
        # image_dir = self.args.image_dir
        # if os.path.isfile(image_dir):
        #     features, infos = self.get_detectron_features([image_dir])
        #     self._save_feature(image_dir, features[0], infos[0])

        paths = os.listdir(self.args.image_dir)

        data = json.load(open('imdb_nuscenes_trainval.json'))['data']
        image_ids = []
        files_list = []
        for idx, datum in enumerate(data):
            print(idx)
            files = {}
            image_names = datum['image_names']
            image_id = datum['image_id']
            image_paths = []
            if image_id not in image_ids:
                image_ids.append(image_id)
                for image_name in image_names:
                    #image_path = [self.args.image_dir + p + 'sample' + image_name + '.jpg' for p in paths if paths[:4] == 'part']
                    image_path = self.arg.image_dir + image_name + '.jpg'
                    image_paths.append(image_path)
                if len(image_names) == len(image_paths) == 6:
                    files_list.extend(image_paths)

        # else:
        #     files = get_image_files(
        #         self.args.image_dir,
        #         exclude_list=self.args.exclude_list,
        #         start_index=self.args.start_index,
        #         end_index=self.args.end_index,
        #         output_folder=self.args.output_folder,
        #     )

        finished = 0
        total = len(files_list)

        for chunk, begin_idx in chunks(files_list, self.args.batch_size):
            features, infos = self.get_detectron_features(chunk)
            for idx, file_name in enumerate(chunk):
                self._save_feature(file_name, features[idx], infos[idx])
            finished += len(chunk)

            if finished % 200 == 0:
                 print(f"Processed {finished}/{total}")


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
