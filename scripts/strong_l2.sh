#!/bin/bash

python -m robustness.main --dataset cifar --eval-only 1 --arch resnet50 --out-dir /tmp/ --constraint 2 --adv-eval 1 --attack-lr 0.0625 --batch-size 128 --attack-steps 20 --eps 0.5 --resume $1
