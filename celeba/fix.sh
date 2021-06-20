#!/bin/bash
for (( i=0; i<=$4; i++ ))
do
    python train_models.py --split $1 --filter $2 --ratio $3 --name $((i+1001))
done

