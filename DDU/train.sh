#! /bin/bash

for i in {1..1}; do
       echo "训练第$i次"
       python train.py \
              --seed 1 \
              --gpu 1 \
              --run 3 \
              --data-aug \
              -b 256 \
              --epochs 350 --dataset cifar10 \
              --model resnet50  \
              -sn
              # -mod \
              # -sn --coeff 3.0 
       sleep 5
done
