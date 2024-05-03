#! /bin/bash

python active_learning_script.py \
       --seed 1 \
       --gpu 0 \
       --model vgg16 \
       -sn -mod \
       --al-type ensemble