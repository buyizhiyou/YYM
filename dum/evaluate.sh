#! /bin/bash
echo "###########  Evaluate with sn and mod;
###########  Usage:
###########   ./evaluate.sh --gpu  0 --run 17 --batchsize 512 --evaltype gmm --ooddataset svhn --model resnet50  --perturbation fgsm --contrastive 0 --adv 0"

# 解析命令行参数
options=$(getopt -o g:r:b:t:d:m:c:a --long gpu:,run:,batchsize:,evaltype:,ooddataset:,model:,perturbation:,contrastive:,adv:, -- "$@")
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
       -t | --evaltype)
              shift
              evaltype=$1
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

#check evaltype
evaltypes=("gmm" "kde" "ensemble")
if [[ " ${evaltypes[@]}" =~ "$evaltype" ]]; then
       echo "###########  evaluate evaltype $evaltype"
else
       echo "###########  $evaltype is not in the evaltypes [gmm,kde,ensemble]"
fi
#check ooddataset
ooddatasets=("tiny_imagenet" "mnist" "lsun" "svhn" "cifar100")
#check model
models=("vgg16" "resnet50" "wide_resnet","vit")
if [[ " ${models[@]}" =~ "$model" ]]; then
       echo "###########  evaluate model $model"
else
       echo "###########  $model is not in the models [vgg16,resnet50,wide_resnet]"
fi

if [[ "$ooddataset" = "all" ]]; then
       for ood in ${ooddatasets[@]}; do
              python evaluate.py \
                     --seed 1 \
                     -b $batchsize \
                     --gpu $gpu \
                     --run $run \
                     --dataset cifar10 \
                     --ood_dataset $ood \
                     --load-path ./saved_models \
                     --model $model \
                     --evaltype $evaltype \
                     --perturbation $perturbation \
                     --contrastive $contrastive \
                     --adv $adv \
                     -mod \
                     -sn
       done
else
       python evaluate.py \
              --seed 1 \
              -b $batchsize \
              --gpu $gpu \
              --run $run \
              --dataset cifar10 \
              --ood_dataset $ooddataset \
              --load-path ./saved_models \
              --model $model \
              --evaltype $evaltype \
              --perturbation $perturbation \
              --contrastive $contrastive \
              --adv $adv \
              -mod \
              -sn
fi
