#!/bin/bash

num=100

if [ ! -d "./logs_taobao" ]; then
    mkdir ./logs_taobao
fi

if [ ! -d "./save_path_taobao" ]; then
    mkdir ./save_path_taobao
fi

rm -f *.csv
rm -f *.pyc
rm -rf logs_taobao/*
rm -rf save_path_taobao/*
rm -rf permanent_answers/
rm -rf taobao_succinct_datasets/

# 开始执行python创建所有worker
python train_ps.py >./logs_taobao/log_0.txt 2>&1 &

sleep 100s

for((i = 1; i <= num; i++))
do
    python train_client.py --machine_index=$i >./logs_taobao/log_$i.txt 2>&1 &
done

echo "Runing $num workers."

exit 0
