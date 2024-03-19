#! /bin/bash

for i in {1..1}; do
       echo "训练第$i次"
       python train.py \
              --seed 1 \
              --gpu 1 \
              --runs 3 \
              --data-aug \
              -b 256 \
              --epochs 300 --dataset cifar10 \
              --model resnet50 \
              -mod \
              -sn --coeff 3.0 

       sleep 5
done
