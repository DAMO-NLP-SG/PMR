#!/bin/bash

hostname
nvidia-smi
task=NER

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12359 $task/train-NER.py \
        --output_dir ./saved_models/conll03-PMR-base \
        --model_type roberta \
        --model_name_or_path ../saved_models/PMR-base --cache_dir ../cache \
        --data_path ./Data/conll03 \
        --do_train --do_eval --do_lower_case \
        --learning_rate 2e-5 \
        --num_train_epochs 15 \
        --per_gpu_eval_batch_size=64  \
        --per_gpu_train_batch_size=64 \
        --max_seq_length 192 --max_query_length 32\
        --save_steps 0 --logging_steps 1000\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training