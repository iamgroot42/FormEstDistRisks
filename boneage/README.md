# RSNA Boneage dataset experiments

## Pre-requisites

- Make sure you have the [RSNA BoneAge dataset](https://www.kaggle.com/kmader/rsna-bone-age) downloaded on your system. Edit `BASE_DATA_DIR` in `data_utils.py` to point to that path.
- Make a directory with the structure: `X/<split>/<ratio>`, and edit `BASE_MODELS_DIR` to `X` in `model_utils.py`.

### Generate data splits

Run this command to generate victim/adversary data splits

`python data_utils.py`

### Feature extraction

Run this command to extract features from images

`python preprocess.py`

## Running experiments

### Training models

Run this command to train 100 models on the adversary's split of data, setting the ratio of females to 0.5. Change arguments to train models while varying ratios

`python train_models.py --ratio 0.5 --split adv --num 100`


### Loss/Threshold tests

Run this command to generate both loss-test and threshold-test numbers for two specified ratios

`python perf_tests.py --ratio_1 0.5 --ratio_2 0.2`


### Meta-classifier

Run this command to train 0.5 v/s X form meta-classifiers on the adversary's models and test them on the victim's models.

`python meta.py --second X`


### Plotting results

Use the following file to plot graphs with experimental results. You can specify arguments to control legend plot, color of plot, and y-axis title (see file for more)

`python make_boxplots.py`