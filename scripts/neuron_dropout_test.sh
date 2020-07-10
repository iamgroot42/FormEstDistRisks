#!/bin/bash
# First argument : source model path
# Second argument : scaling factor
# Third argument : N
echo "Running $4 runs"
for(( i = 1; i<= $4; i++))
do
  echo "<== Run $i"
  cp $1 /p/adversarialml/as9rw/models_cifar10_densenet/delta_model.pt
  python delta_defense.py $2 $3
  python -m robustness.main --dataset cifar --eval-only 1 --out-dir dudu --arch densenet169 --adv-eval 1 --resume /p/adversarialml/as9rw/models_cifar10_densenet/delta_model.pt --constraint 2 --attack-steps 20 --eps 0.5 --attack-lr 0.0625 --batch-size 64
  python -m robustness.main --dataset cifar --eval-only 1 --out-dir dudu --arch densenet169 --adv-eval 1 --resume /p/adversarialml/as9rw/models_cifar10_densenet/delta_model.pt --constraint inf --attack-steps 20 --eps 0.031372549 --attack-lr 0.00392156863 --batch-size 64
  echo "Run Complete ==>"
done