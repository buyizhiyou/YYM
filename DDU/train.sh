#! /bin/bash

for i in {1..2}; do
       echo "训练第$i次"
       python train.py \
              --seed 1 \
              --gpu 0 \
              --run 8 \
              --data-aug \
              -b 256 \
              --epochs 10 --dataset cifar10 \
              --model vgg16  \
              -mod \
              -sn
       sleep 5
done
