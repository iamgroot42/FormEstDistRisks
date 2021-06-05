#!/bin/bash
for i in {1..1000}
do
    python train_models.py --split $1 --filter $2 --ratio $3 --name $i
done
