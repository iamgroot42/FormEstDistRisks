# ogbn-arxiv Boneage dataset experiments

## Pre-requisites

- Make a directory with the structure: `X/<split>/deg_Y` for degree `Y`, and edit `BASE_MODELS_DIR` to `X` in `model_utils.py`.

## Running experiments

### Training models

Run this command to train a model on the adversary's split of data, setting the desired mean node degree to 13, and save model at filpath `Z` . Change arguments to train models while varying degrees. By default, 1% of the nodes will be pruned randomly per experiment.

`python train_models.py --degree 13 --split adv --savepath Z --gpu`

To train 1000 together, run `bulk_deg.sh adv 13`


### Loss/Threshold tests

Run this command to generate both loss-test and threshold-test numbers for the specified degree for 13 v/s X (12, in the case below)

`python perf_tests.py --gpu --deg 12`


### Meta-classifier

Run this command to train 12 v/s 13 form meta-classifiers on the adversary's models and test them on the victim's models.

`python meta.py --degrees 12,13 --gpu --parallel`

To train the regression variant of the meta-classifier, run the following:

`python meta_regress.py --regression`


### Meta-classifier test

To test the trained meta-classifier on all distributions, run:

`python meta_regress_test.py --gpu`

### Plotting results

Use the following file to plot graphs with experimental results. You can specify arguments to control legend plot, color of plot, and y-axis title (see file for more)

`python make_boxplots.py --filter race`