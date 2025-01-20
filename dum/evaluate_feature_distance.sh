#! /bin/bash
echo "###########  Evaluate with sn and mod;
###########  Usage:
###########   ./evaluate_feature_distance.sh --gpu  0 --run 38 --batchsize 512 --model resnet50  --perturbation fgsm --contrastive 0 --adv 0 --size 32"

# 解析命令行参数
options=$(getopt -o g:r:b:t:d:m:c:a:s --long gpu:,run:,batchsize:,model:,perturbation:,contrastive:,adv:,size:,  -- "$@")
eval set -- "$options"
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
       -c | --contrastive)
              shift
              contrastive=$1
              shift
              ;;
       -a | --adv)
              shift
              adv=$1
              shift
              ;;
       -s | --size)
              shift
              size=$1
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
if [[ -z "$gpu" ]]; then
       echo "###########  Error: parameter gpu is required"
       exit 1
fi



#check model
models=("vgg16" "resnet50" "resnet18" "wide_resnet","vit")
if [[ " ${models[@]}" =~ "$model" ]]; then
       echo "###########  evaluate model $model"
else
       echo "###########  $model is not in the models [vgg16,resnet50,wide_resnet]"
fi


python evaluate_feature_distance.py \
       --seed 1 \
       -b $batchsize \
       --gpu $gpu \
       --run $run \
       --dataset cifar10 \
       --load-path ./saved_models \
       --model $model \
       --perturbation $perturbation \
       --contrastive $contrastive \
       --adv $adv \
       --size $size \
       -mod \
       -sn
