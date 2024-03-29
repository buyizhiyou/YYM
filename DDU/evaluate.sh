#! /bin/bash

python evaluate.py \
       --seed 1 \
       -b 64 \
       --gpu 0 \
       --run 2 \
       --dataset cifar10 \
       --ood_dataset svhn \
       --load-path ./saved_models \
       --model resnet50 \
       --model-type gmm \
       -mod \
       -sn --coeff 3.0