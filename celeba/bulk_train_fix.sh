#!/bin/bash
for (( i = 1001; i <= 1000+$4; i++ )) 
do
    python train_models.py --split $1 --filter $2 --ratio $3 --name ${i} --task Male
done
