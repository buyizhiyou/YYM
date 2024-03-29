#!/bin/bash

func() {
    echo "Usage:"
    echo "train_randompatch_cifar10.sh [-i CFG] "
    echo "Description:"
    echo "CFG,the path of config file"
    exit -1
}


while getopts 'h:i:u' OPT; do
    case $OPT in
        i) CFG="$OPTARG";;
        u) upload="true";;
        h) func;;
        ?) func;;
    esac
done


echo "train $CFG"
python train_classification_randompatch.py --config $CFG
