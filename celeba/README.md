# CelebA dataset experiments

## Dataset

You can use the exact victim-adversary splits from [here](https://github.com/iamgroot42/FormEstDistRisks/blob/main/celeba/data/celeba_splits.zip)

## Pre-requisites

- Make sure you have the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) downloaded on your system. Edit `BASE_DATA_DIR` in `data_utils.py` to point to that path.
- Make a directory with the structure: `X/<split>/<attribute>/<ratio>`, and edit `BASE_MODELS_DIR` to `X` in `model_utils.py`.

### Generate data splits

Run this command to generate victim/adversary data splits (or use the ones provided by us to recreate experiments exactly)

`python data_utils.py`

## Running experiments

### Training models

Run this command to train a model on the adversary's split of data, setting the ratio of males to 0.5. Change arguments to train models while varying ratios

`python train_models.py --ratio 0.5 --split adv --filter Male --num 100`

For `female`, we observed a test accuracy of around 90% on the adversary's split of data, and 91% on the victim's split of data.
For `old`, we observed a test accuracy of around 94% on the adversary's split of data, and 95% on the victim's split of data.

### Loss/Threshold tests

Run this command to generate both Loss-Test (Sec 5.1.1) and Threshold-Test (Sec 5.1.2) numbers for two specified ratios

`python perf_tests.py --ratio_1 0.5 --ratio_2 0.2 --filter Male`

### Meta-classifier

Run this command to train 0.5 v/s X form meta-classifiers (Sec 5.2.1) on the adversary's models and test them on the victim's models.

`python meta.py --second X --filter Male`

Run this command to train regression meta-classifiers (Sec 6.4)

`python meta_regression.py --second X --filter Male`

Use `--eval_only` mode with a trained regression meta-classifier to perform binary classification.

Results observed across all these attacks should roughly match (within margin of error) the ones observed in Figures 2(c), 3(b), 4, & 7.
