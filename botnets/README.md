# BotNet-Classification

### Setting it up

Install a fork of the `botnet-detection` repository:

```bash
git clone https://github.com/iamgroot42/botnet-detection.git
cd botnet-detection
python setup.py install
```

## Dataset

Split files for victim/adversary are already provided in the repository (and automatically). You can download the processed and parsed version of the dataset [here](https://archive.org/details/FormEstDistRisks_botnet)

## Running experiments

### Training models

Run this command to train a model on the adversary's split of data, setting the desired property to 1 (coeff > 0.0071), and save model at filpath `Z`

`python train_models.py --prop_val 1 --split adv --savename Z --gpu`

We observed a test accuracy of around 92% on the adversary's split of data, and 89% on the victim's split of data.

### Loss/Threshold tests

Run this command to generate both lLoss-Test (Sec 5.1.1) and Threshold-Test (Sec 5.1.2) numbers.

`python perf_tests.py --gpu`

### Meta-classifier

Run this command to train a meta-classifiers (Sec 5.2.1) on the adversary's models and test them on the victim's models.

`python meta.py --gpu --parallel`

Results observed across all these attacks should roughly match (within margin of error) the ones reported in Section 6.5
