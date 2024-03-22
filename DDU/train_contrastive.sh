#! /bin/bash

#! /bin/bash

for i in {1..1}; do
       echo "训练第$i次"
       python train.py \
              --seed 1 \
              --gpu 1 \
              --run 3 \
              --scheduler cos \
              -b 128 \
              --epochs 1300 --dataset cifar10 \
              --model resnet50 \
              -mod \
              -sn --coeff 3.0 \
              --contrastive 2

       sleep 5
done
