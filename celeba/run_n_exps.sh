#!/bin/bash

ratios=(0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0)

for ratio in ${ratios[@]}; do
    python meta.py --n_tries 10 --train_sample 5 --second ${ratio} --first_n_conv 4 --focus conv
    python meta.py --n_tries 10 --train_sample 10 --second ${ratio} --first_n_conv 4 --focus conv
    python meta.py --n_tries 10 --train_sample 20 --second ${ratio} --first_n_conv 4 --focus conv
    python meta.py --n_tries 10 --train_sample 1600 --second ${ratio} --first_n_conv 4 --focus conv
done

