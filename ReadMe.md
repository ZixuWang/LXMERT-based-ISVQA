# LXMERT based ISVQA in autonomous driving dataset(Nuscenes)
This repository is to implement lxmert model based VQA in autonomous driving dataset(Nuscenes) 

# Dataset introduction
[ToDo] blabla
```sh
|-- extracted_features
|   |-- train
|   |-- test
|-- mini
|   |-- maps
|   |   |-- 36092f0b03a857c6a3403e25b4b7aab3.png
|   |-- samples
|   |   |-- CAM_BACK
|   |   `-- ....
|   `-- v1.0-mini
|       |-- attribute.json
|       `-- ...
|-- part01
|   `-- samples
|       |-- CAM_BACK
|       `-- ...
|-- part02
|   `-- samples
|       |-- CAM_BACK
```

# File structure
```sh
-- input
  -- ISVQA
    -- NuScenes
      -- extracted_features
        -- test
        -- train
    -- jsons
  -- ProcessedFile
-- output
-- others(temporary)
-- src
|   -- DataPreproScript
|   -- lxrt
|   -- pretrain
      --model_LXRT.pth
|   -- param.py
|   -- vqa_data.py
|   -- vqa_model.py
-- AddLabel.py
-- ReadMe.md
-- feature_extaction.py
-- data_preprocessing.py
-- ISVQA_main.py
```


# Preparation
### Prerequisite
conda install -c anaconda boto3  # TODO
- MMF install [Official instruction](https://mmf.sh/docs/), download the mmf repo under '/src'.
```
cd src
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable .  # after this one, it will become pytorch 1.9 automatically
```
- Mask-RCNN backbone [instruction](https://mmf.sh/docs/tutorials/image_feature_extraction/), download the repo under '/src'.
```
pip install ninja yacs cython matplotlib
pip install opencv-python
cd src
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build develop
```
- You might have such a following bug, change PY3 to PY37:
File "/root/Documents/ISVQA/src/vqa-maskrcnn-benchmark/maskrcnn_benchmark/utils/imports.py", line 4, in <module>
    if torch._six.PY3:
AttributeError: module 'torch._six' has no attribute 'PY3'
```
- LXMERT repository [instruction](https://github.com/airsplay/lxmert/blob/master/requirements.txt) 
- download pretrained lxmert model via
```sh
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```
- Download all_ans.json from https://github.com/airsplay/lxmert/blob/master/data/lxmert/all_ans.json for pretrained model and save it under input/ProcessedFile
- maskrcnn_benchmark
- mmf (orinially is ..., mmf is too large ...)
- cv
- python version, pytorch version

# Feature extraction
```sh
python feature_extraction.py
```

# Question ID and score generation
For original question id is not unique for each question, we need to generate new question id to identify each question.
Besides, we also need to generate answer score for each answer.

You can either directly use preprocessed annotation file(*trainval_with_score_quesid.json* for training and *test_with_score_quesid.json* for testing) and *ans2label.txt* under *input/ProcesseFile* or do it on your own following below steps.
1. Download the original annotation files and answer file from [ISVQA](https://github.com/ankanbansal/ISVQA-Dataset/tree/master/nuscenes)
2. Generate new annotation file and answer file via *data_preprocessing.py*. Before running *data_preprocessing.py*, don't forget to change file path to yours.
3. Run *AddLabel.py* to generate new *ans2label.txt* 
4. Remember to name it as *trainval_with_score_quesid.json*, *test_with_score_quesid.json*, *ans2label.txt* and put them under *input/ProcesseFile*


# Training and Test
When features, new .json file and .txt file are ready, run *ISVQA_main.py* to train and test the whole model.
```sh
python ISVQA_main.py
```

Notice that to save time the default setting is train and test on small dataset, if you want to train and test on the whole dataset, please uncomment the line 139 - 141.


# Result
After 100 Epochs, we have the accuracy on training set as xxx and on test set as xxx.

*figure1*

*figure2*

*figure3*

