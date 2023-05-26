# til-23-cv

> DSTA BrainHack TIL 2023 Qualifiers Computer Vision Task Code

## Installation

Only Python 3.9 is supported at the moment to speed up dependency resolution.

```sh
git clone --recurse-submodules https://github.com/Interpause/til-23-cv.git
pip install -r requirements.txt

# Or...
# TODO: Support below; Key requirement is to code with module invocation in mind
pip install git+https://github.com/Interpause/til-23-cv.git
```

## Training

### Object Detection

We rely on [Ultralytics](https://github.com/ultralytics/ultralytics) to train YOLO variants. See [`yolo.ipynb`](./notebooks/yolo.ipynb) for use the command below:

```sh
yolo detect train cfg=cfg/yolov5x6u_mvp.yaml
```

## Contribution Guide

- We use [Black](https://github.com/psf/black) for formatting, [isort](https://github.com/PyCQA/isort) for import sorting, and Google-style docstrings.
- To generate `requirements.txt`, `poetry export -f requirements.txt -o requirements.txt --without-hashes`.
