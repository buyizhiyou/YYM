#! /bin/bash
echo "Usage ./evaluate.sh --gpu  0 --run 6 --batchsize 64 --ooddataset svhn --model  resnet50 --contrastive 0"
# 解析命令行参数
options=$(getopt -o g:r:b:d:m:c --long gpu:,run:,batchsize:,ooddataset:,model:,contrastive:, -- "$@")
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
       -d | --ooddataset)
              shift
              ooddataset=$1
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

python evaluate.py \
       --seed 1 \
       -b $batchsize \
       --gpu $gpu \
       --run $run \
       --dataset cifar10 \
       --ood_dataset $ooddataset \
       --load-path ./saved_models \
       --model $model \
       --model-type gmm \
       --contrastive $contrastive \
       -mod \
       -sn