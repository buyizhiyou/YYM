#!/bin/bash

python main.py -a vgg16 --dist-url 'tcp://127.0.0.1:3000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 \
            --rank 0   /share/home/shiqing/YYM/data/imageNet