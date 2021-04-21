#!/bin/bash

declare -a trainsets=("cifar10" "cifar100" "svhn")
declare -a methods=("baseline" "odin" "gram")
declare -a models=("densenet" "resnet")

cd ../src || exit

for trainset in ${trainsets[@]}; do
  for method in ${methods[@]}; do
    for model in ${models[@]}; do
      echo $method $model $trainset


      if [ $method == "odin" ]; then
          batch_size=256
      else
          batch_size=2048
      fi


      python main_execute_method.py method=$method model=$model trainset=$trainset batch_size=$batch_size
    done
  done
done

# Energy method
method="energy"
declare -a trainsets=("cifar10" "cifar100")
model="wrn"

for trainset in ${trainsets[@]}; do
  echo $method $model $trainset
  python main_execute_method.py method=$method model=$model trainset=$trainset
done