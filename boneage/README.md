# RSNA Boneage dataset experiments

## Dataset

You can use the files [here](https://github.com/iamgroot42/FormEstDistRisks/raw/main/boneage/data/boneage_splits.zip) - these contain split information for victim/adversary, as well as processed features that we used in our experiments.

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

We observed a test accuracy of around 83% on the adversary's split of data, and 81% on the victim's split of data.

### Loss/Threshold tests

Run this command to generate both Loss-Test (Sec 5.1.1) and Threshold-Test (Sec 5.1.2) numbers for two specified ratios

`python perf_tests.py --ratio_1 0.5 --ratio_2 0.2`

### Meta-classifier

Run this command to train 0.5 v/s X form meta-classifiers (Sec 5.2.1) on the adversary's models and test them on the victim's models.

`python meta.py --second X`

Run this command to train regression meta-classifiers (Sec 6.4)

`python meta_regression.py --second X`

Use `--eval_only` mode with a trained regression meta-classifier to perform binary classification.

### Plotting results

Use the following file to plot graphs with experimental results. You can specify arguments to control legend plot, color of plot, and y-axis title (see file for more)

`python make_boxplots.py`

Results observed across all these attacks should roughly match (within margin of error) the ones observed in Figures 2(b), 5(b), & 6(b).
