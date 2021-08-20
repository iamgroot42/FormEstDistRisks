#!/bin/bash

python meta.py --first_n 1 --filter sex --train_sample 5
python meta.py --first_n 1 --filter sex --train_sample 10
python meta.py --first_n 1 --filter sex --train_sample 20
python meta.py --first_n 1 --filter sex --train_sample 1600
