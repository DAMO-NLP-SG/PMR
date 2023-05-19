#!/bin/bash

hostname
nvidia-smi
task=GLUE

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12358 $task/train-GLUE.py \
        --output_dir ./saved_models/glue-PMR-large \
        --model_type roberta \
        --model_name_or_path ../saved_models/PMR-large --cache_dir ../cache \
        --data_path SST-2 \
        --do_train --do_eval --do_lower_case \
        --learning_rate 2e-5  \
        --num_train_epochs 2 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=16 \
        --max_seq_length 192 --max_query_length 64\
        --save_steps 0 --logging_steps 2000\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training