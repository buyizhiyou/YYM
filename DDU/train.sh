#! /bin/bash

echo "Usage ./train.sh --gpu  0 --run 8 --batchsize 64 --epochs 300 --model  vgg16 --contrastive 0"
# 解析命令行参数
options=$(getopt -o g:r:b:e:m:c --long gpu:,run:,batchsize:,epochs:,model:,contrastive:, -- "$@")
eval set -- "$options"

# 提取选项和参数
while true; do
       case $1 in
       -g | --gpu)
              shift
              gpu=$1
              shift
              ;;
       -r | --run)
              shift
              run=$1
              shift
              ;;
       -b | --batchsize)
              shift
              batchsize=$1
              shift
              ;;
       -e | --epochs)
              shift
              epochs=$1
              shift
              ;;
       -m | --model)
              shift
              model=$1
              shift
              ;;
       -c | --contrastive)
              shift
              contrastive=$1
              shift
              ;;
       --)
              shift
              break
              ;;
       # *) echo "Invalid option: $1" exit 1 ;;
       esac
done

# 检查变量
if [ -z "$gpu" ]; then
       echo "Error: parameter gpu is required"
       exit 1
fi

for i in {1..10}; do
       echo "训练第$i次"
       python train.py \
              --seed 1 \
              --gpu $gpu \
              --run $run \
              --data-aug \
              -b $batchsize \
              --epochs $epochs --dataset cifar10 \
              --model $model \
              --contrastive $contrastive \
              -mod \
              -sn
       sleep 5
done
