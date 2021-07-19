#!/bin/bash
for i in {1..500}
do
    python train_models.py --gpu --split $1 --prop_val $2 --savename $i --best_model
done
