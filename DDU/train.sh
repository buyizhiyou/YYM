#! /bin/bash

for i in {1..1}; do
       echo "训练第$i次"
       python train.py \
              --seed 1 \
              --gpu 1 \
              --runs 3 \
              -b 64 \
              --epochs 300 --dataset cifar10 \
              --model resnet50 \
              -mod \
              -sn --coeff 3.0 \
              --contrastive true

       sleep 5
done