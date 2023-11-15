#!/bin/bash

lr=1.0
batch_size=256
epochs=100
mode='train'
save_dir='./saved_models/bayesian'

python examples/main_bayesian_mnist.py --lr=$lr --batch-size=$batch_size --epochs=$epochs --mode=$mode --save_dir=$save_dir

