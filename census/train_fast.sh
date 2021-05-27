#!/bin/bash

N_PER_JOB=500
N_JOBS=2
for ((i=0;i<N_JOBS;i++)); do
    OFFSET=$((N_PER_JOB * i + 0))
    python train_models.py --filter $2 --num $N_PER_JOB --split $1 --ratio $3 --offset $OFFSET &
done

wait
echo "Trained and saved all models!"
