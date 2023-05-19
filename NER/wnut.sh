#!/bin/bash

hostname
nvidia-smi

task=NER
CUDA_VISIBLE_DEVICES=0 python  $task/train-NER.py \
        --output_dir ./saved_models/wnut-PMR-base \
        --model_type roberta \
        --model_name_or_path ../saved_models/PMR-base --cache_dir ../cache \
        --data_path ./Data/wnut \
        --do_train --do_eval --do_lower_case \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --per_gpu_eval_batch_size=64  \
        --per_gpu_train_batch_size=32 \
        --max_seq_length 160 --max_query_length 32\
        --save_steps 0 --logging_steps 1000\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training