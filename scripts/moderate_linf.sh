#!/bin/bash

python -m robustness.main --dataset cifar --eval-only 1 --arch resnet50 --out-dir /tmp/ --constraint inf --adv-eval 1 --attack-lr 0.00392156862 --batch-size 128 --attack-steps 20 --eps 0.031372549 --resume $1
