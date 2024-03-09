#! /bin/bash

python pretrain_test.py \
       --seed 1 \
       --dataset cifar10 \
       --ood_dataset svhn \
       --load-path ./saved_models \
       --model resnet50 \
       --model-type gmm