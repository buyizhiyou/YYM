#! /bin/bash

python train.py \
       --seed 1 \
       -b 256 \
       --dataset cifar10 \
       --model resnet50 \
       -mod  \
        -sn --coeff 3.0 