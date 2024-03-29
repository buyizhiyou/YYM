#! /bin/bash

python evaluate.py \
       --seed 1 \
       -b 64 \
       --gpu 1 \
       --run 2 \
       --dataset cifar10 \
       --ood_dataset svhn \
       --load-path ./saved_models \
       --model resnet50 \
       --model-type softmax \
       --contrastive 1 \
       -mod \
       -sn --coeff 3.0