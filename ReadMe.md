# LXMERT based ISVQA in autonomous driving dataset(Nuscenes)
This repository is to implement lxmert model based VQA in autonomous driving dataset(Nuscenes) 

# Dataset introduction

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
  # -- pretrain-weights
  -- json
-- output
-- others(temporary)
-- src
|   -- lxrt
|   -- maskrcnn benchmark
|   -- vqa_data.py
|   -- utils
-- ReadMe.md
-- feature_extaction.py
-- data_preprocessing.py
-- ISVQA_main.py
```


# Preparation
- Install mask rcnn benchmark via [instruction](https://mmf.sh/docs/tutorials/image_feature_extraction/) to extract image features
- Install the environment to run LXMERT model via [instruction](https://github.com/airsplay/lxmert/blob/master/requirements.txt) 
- Download pretrained lxmert model via
```sh
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P src/pretrain
```

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


# Training and Test
When features, new .json file and .txt file are ready, run *ISVQA_main.py* to train and test the whole model.
```sh
python ISVQA_main.py
```

# Result
After 100 Epochs, we have the accuracy on training set as xxx and on test set as xxx.
*figure1*
*figure2*
*figure3*

