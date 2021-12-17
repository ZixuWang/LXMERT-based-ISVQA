# ISVQA

introduction....

# Dataset intro
blabla
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
-- others
-- lxrt
-- src
|   -- lxrt
|   -- maskrcnn benchmark
|   -- vqa_data.py
|   -- utils
-- ReadMe.md
-- feature_extaction.py
-- json_generation.py
-- ISVQA_main.py
```


# Preparation 
- 1. Install mask rcnn benchmark via [instruction](https://mmf.sh/docs/tutorials/image_feature_extraction/) to extract image features
- 2. Install the environment to run LXMERT model [instruction](https://github.com/airsplay/lxmert/blob/master/requirements.txt) 
- 2. download pretrained lxmert model via
```sh
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P src/pretrain
```

# Feature extraction
run 
```sh
python feature_extaction.py
```

# ID and score generation
```sh
python data_preprocessing.py
```

# Training and Test
```sh
python ISVQA_main.py
```

# Result

