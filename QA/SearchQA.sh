#!/bin/bash

hostname
nvidia-smi
task=QA

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12358  $task/train-QA.py \
        --output_dir ./saved_models/mrqa-PMR-base \
        --model_type roberta \
        --model_name_or_path ../saved_models/PMR-base --cache_dir ../cache \
        --data_path SearchQA \
        --do_train --do_eval --do_lower_case \
        --learning_rate 2e-5 \
        --num_train_epochs 4 \
        --per_gpu_eval_batch_size=16  \
        --per_gpu_train_batch_size=16 \
        --max_seq_length 384 --max_query_length 64\
        --save_steps 0 --logging_steps 5000\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training