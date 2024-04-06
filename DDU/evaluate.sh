#! /bin/bash

python evaluate.py \
       --seed 1 \
       -b 64 \
       --gpu 0 \
       --run 5 \
       --dataset cifar10 \
       --ood_dataset lsun \
       --load-path ./saved_models \
       --model resnet50 \
       --model-type gmm \
       -mod \
       -sn --coeff 3.0 \
       --mcdropout true