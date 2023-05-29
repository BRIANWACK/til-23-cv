# til-23-cv

> DSTA BrainHack TIL 2023 Qualifiers Computer Vision Task Code

## Notes for Judges

### Object Detection

- YOLOv5u6 models were chosen because `1280p` pretraining, see <https://docs.ultralytics.com/models/yolov5>.
  - Model download: <https://github.com/ultralytics/assets/releases/tag/v0.0.0>.
  - `ultralytics` auto-download: `model = YOLO("yolov5m6u.pt")`, `yolo detect benchmark model=yolov5m6u.pt`, or any `yolo` command with `model=yolov5m6u.pt`.
  - `yolov5m6u.pt` was chosen based on vram limitations and effective batch size.
- Model training was stopped early at epoch 84, and `best.pt` was renamed to `custom_yolov5m6u_best.pt`.
  - NOTE: Config file specifies 300 epochs to preserve original lr curve used. `Ctrl+C` recommended to interrupt training early.
  - See <https://github.com/ultralytics/ultralytics/blob/dada5b7/ultralytics/yolo/utils/metrics.py#L657-L660> for definition of best model.
- For modifications made to [`ultralytics`](https://github.com/ultralytics/ultralytics), see <https://github.com/Interpause/ultralytics/compare/f23a035...main>.
  - Notably, `ultralytics` hardcodes [Albumentations](https://albumentations.ai/).
  - I wasn't going to wait for a PR so I just modified their hardcoded stuff.

### Suspect Recognition

- DINOv2 model was used as encoder/backbone via [`timm`](https://github.com/huggingface/pytorch-image-models).
  - Model download: <https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m>.
  - DINOv2 (ViT-S/14) chosen for similar size to ResNet50 and State of the Art Object-Centric Representations via Self-Supervised Learning.
- TODO: Reproducible hyperparameter system.
- ArcFace loss was used to reshape model's latent space into a hypersphere for cosine similarity: <https://arxiv.org/abs/1801.07698>.

### Others

- "I have code OCD. Poetry is love, Poetry is life. And GG users of Conda."
  - As one said in a group chat by the great engineer, [@Interpause](https://github.com/interpause) (John-Henry Lim).
- The real reason I chose DINOv2 was PTSD.
  - Behold! My internship project "Self-Supervised Learning of Video Object Segmentation using DINOSAUR and SAVi": [Interpause/dinosavi](https://github.com/Interpause/dinosavi).

## Installation

Only Python 3.9 is supported at the moment to speed up dependency resolution.

```sh
git clone --recurse-submodules https://github.com/Interpause/til-23-cv.git
pip install -r requirements.txt

# Or...
# TODO: Support below; Key requirement is to code with module invocation in mind
pip install git+https://github.com/Interpause/til-23-cv.git
```

Or if in a notebook:

```py
!git submodule update --init --recursive
%pip install -r requirements.txt
# May be ../ultralytics if magic cwd cell isn't run yet.
%pip install -e ./ultralytics
```

## Data Preparation

- Download & extract the following files from <https://zindi.africa/competitions/brainhack-til-23-advanced-cv/data>:
  - `Train.zip` -> `data/til23plush/images/train/*.png`
  - `Validation.zip` -> `data/til23plush/images/val/*.png`
  - `Test.zip` -> `data/til23plush/images/test/*.png`
  - `train_labels.zip` -> `data/til23plush/labels/train/*.txt`
  - `val_labels.zip` -> `data/til23plush/labels/val/*.txt`
  - `suspects.zip` -> `data/til23plush/suspects/*.png`
- Write `data/til23plush/dataset.yaml`:

  ```yaml
  train: images/train
  val: images/val
  test: images/test
  nc: 200
  names: ['#0','#1','#2','#3','#4','#5','#6','#7','#8','#9','#10','#11','#12','#13','#14','#15','#16','#17','#18','#19','#20','#21','#22','#23','#24','#25','#26','#27','#28','#29','#30','#31','#32','#33','#34','#35','#36','#37','#38','#39','#40','#41','#42','#43','#44','#45','#46','#47','#48','#49','#50','#51','#52','#53','#54','#55','#56','#57','#58','#59','#60','#61','#62','#63','#64','#65','#66','#67','#68','#69','#70','#71','#72','#73','#74','#75','#76','#77','#78','#79','#80','#81','#82','#83','#84','#85','#86','#87','#88','#89','#90','#91','#92','#93','#94','#95','#96','#97','#98','#99','#100','#101','#102','#103','#104','#105','#106','#107','#108','#109','#110','#111','#112','#113','#114','#115','#116','#117','#118','#119','#120','#121','#122','#123','#124','#125','#126','#127','#128','#129','#130','#131','#132','#133','#134','#135','#136','#137','#138','#139','#140','#141','#142','#143','#144','#145','#146','#147','#148','#149','#150','#151','#152','#153','#154','#155','#156','#157','#158','#159','#160','#161','#162','#163','#164','#165','#166','#167','#168','#169','#170','#171','#172','#173','#174','#175','#176','#177','#178','#179','#180','#181','#182','#183','#184','#185','#186','#187','#188','#189','#190','#191','#192','#193','#194','#195','#196','#197','#198','#199']
  ```

### Object Detection

- See [`notebooks/data.ipynb`](./notebooks/data.ipynb) for converting `til23plush` into `til23plushonly` dataset, where all classes are relabelled to `plushie`.

### Suspect Recognition

- See [`notebooks/data.ipynb`](./notebooks/data.ipynb) for converting `til23plush` into `til23reid` dataset, which uses the typical image classification folder structure.

## Training

### Object Detection

See [`notebooks/yolo.ipynb`](./notebooks/yolo.ipynb) or use the command below:

```sh
yolo detect train cfg=cfg/custom.yaml model=yolov5m6u.pt data=data/til23plushonly/dataset.yaml workers=8 batch=8
```

### Suspect Recognition

See [`notebooks/reid.ipynb`](./notebooks/reid.ipynb).

TODO: CLI and hyperparameter system for training.

## Inference

See [`notebooks/infer.ipynb`](./notebooks/infer.ipynb).

## Contribution Guide

- We use [Black](https://github.com/psf/black) for formatting, [isort](https://github.com/PyCQA/isort) for import sorting, and Google-style docstrings.
- To generate `requirements.txt`, `poetry export -f requirements.txt -o requirements.txt --without-hashes`.
