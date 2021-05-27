# Census dataset experiments

## Pre-requisites

- Make sure you have the adult dataset downloaded on your system. Edit `BASE_DATA_DIR` in `data_utils.py` to point to that path.
- Make a directory with the structure: `X/<split>/<filter>/<ratio>`, and edit `BASE_MODELS_DIR` to `X` in `model_utils.py`.

## Running experiments

### Training models

Run this command to train 100 models on the adversary's split of data, setting the ratio of females to 0.5. Change arguments to train models while varying other attributes and ratios

`python train_ratio_models.py --filter sex --ratio 0.5 --split adv --num 100`

Alternatively, you can run `train_fast.sh adv sex 0.5` to generate 1000 models (via 2 parallel processes).


### Loss/Threshold tests

Run this command to generate both loss-test and threshold-test numbers for two specified ratios for the specified attribute

`python perf_tests.py --filter race --ratio_1 0.5 --ratio_2 0.2`


### Meta-classifier

Run this command to train 0.5 v/s X form meta-classifiers on the adversary's models and test them on the victim's models. Use 1000 epochs for sex, 500 for race.

`python meta.py --filter sex`


### Plotting results

Use the following file to plot graphs with experimental results. You can specify arguments to control legend plot, color of plot, and y-axis title (see file for more)

`python make_boxplots.py --filter race`