#! /bin/bash
echo "Usage ./active_learning.sh --gpu  0 --type ensemble --dataset mnist --model  resnet18  --perturbation 0 "
# 解析命令行参数
options=$(getopt -o g:t:d:m:p --long gpu:,type:,dataset:,model:,perturbation:, -- "$@")
eval set -- "$options"

# 提取选项和参数
while true; do
       case $1 in
       -g | --gpu)
              shift
              gpu=$1
              shift
              ;;
       -t | --type)
              shift
              type=$1
              shift
              ;;
       -d | --dataset)
              shift
              dataset=$1
              shift
              ;;
       -m | --model)
              shift
              model=$1
              shift
              ;;
       -p | --perturbation)
              shift
              perturbation=$1
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

# 检查变量
if [ -z "$perturbation" ]; then
       echo "Error: parameter perturbation is required"
       exit 1
fi

python active_learning_script.py \
       --seed 1 \
       --gpu $gpu \
       --model $model \
       --perturbation $perturbation \
       -sn -mod \
       --al-type $type