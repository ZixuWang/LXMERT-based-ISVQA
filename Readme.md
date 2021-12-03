# ISVQA

introduction....

# Dataset intro
blabla
```sh
-- feature_output_test_test_test 
|-- feature_output_train_test
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
-- vqa_vqa(isvqa_main.py)
-- requirement.yaml
```


# Preparation 
- mask rcnn benchmark [instruction](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark) 
- LXMERT repo
pip-requirement.yaml
- maskrcnn_benchmark
- mmf (orinially is ..., mmf is too large ...)
- cv
- python version, pytorch version

pip install yaml

# Feature extraction

# ID and score generation

# Training
```sh
python vqa_vqa.py
```

# Test
```sh
python vqa.py
```

# Training and result

