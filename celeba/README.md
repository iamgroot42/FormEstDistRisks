# CelebA dataset experiments

## Pre-requisites

- Make sure you have the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) downloaded on your system. Edit `BASE_DATA_DIR` in `data_utils.py` to point to that path.
- Make a directory with the structure: `X/<split>/<attribute>/<ratio>`, and edit `BASE_MODELS_DIR` to `X` in `model_utils.py`.

### Generate data splits

Run this command to generate victim/adversary data splits

`python data_utils.py`

## Running experiments

### Training models

Run this command to train a model on the adversary's split of data, setting the ratio of males to 0.5. Change arguments to train models while varying ratios

`python train_models.py --ratio 0.5 --split adv --filter Male --num 100`


### Loss/Threshold tests

Run this command to generate both loss-test and threshold-test numbers for two specified ratios

`python perf_tests.py --ratio_1 0.5 --ratio_2 0.2 --filter Male`


### Meta-classifier

Run this command to train 0.5 v/s X form meta-classifiers on the adversary's models and test them on the victim's models.

`python meta.py --second X --filter Male`

Run this command to train regression meta-classifiers

`python meta_regression.py --second X --filter Male`

Use `--eval_only` mode with a trained regression meta-classifier to perform binary classification.
