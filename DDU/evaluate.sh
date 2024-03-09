#! /bin/bash

python evaluate.py \
       --seed 1 \
       -b 16 \
       --dataset cifar10 \
       --ood_dataset cifar100 \
       --load-path ./saved_models \
       --model resnet50 \
       --model-type gmm \
       -mod \
       -sn --coeff 3.0 