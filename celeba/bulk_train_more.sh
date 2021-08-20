#!/bin/bash
for i in {1001..2000}
do
    python train_models.py --split $1 --filter $2 --ratio $3 --name $i --task Male
done
