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
  -- image
    -- feature
  -- pretrain-weights
  -- json
-- output
-- lxrt
-- src
|   -- lxrt
|   -- maskrcnn benchmark
|   -- vqa_data_.py
|   -- utils
-- ReadMe.md
-- feature_extaction.py
-- json_generation.py
-- ISVQA_main.py
-- requirement.yaml
```


# Preparation 
- mask rcnn benchmark [instruction](https://mmf.sh/docs/tutorials/image_feature_extraction/) 
- LXMERT repository [instruction](https://github.com/airsplay/lxmert/blob/master/requirements.txt) 
- download pretrained lxmert model via
```sh
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```

- maskrcnn_benchmark
- mmf (orinially is ..., mmf is too large ...)
- cv
- python version, pytorch version

pip install yaml

# Feature extraction
run 
```sh
python feature_extaction.py
```

# ID and score generation


# Training
```sh
python ISVQA_main.py
```

# Test
```sh
python vqa.py
```

# Training and result

