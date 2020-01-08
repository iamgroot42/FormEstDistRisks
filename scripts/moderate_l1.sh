#!/bin/bash

python -m robustness.main --dataset cifar --eval-only 1 --arch resnet50 --out-dir /tmp/ --constraint 1 --adv-eval 1 --attack-lr 1 --batch-size 128 --attack-steps 20 --eps  10 --resume $1
