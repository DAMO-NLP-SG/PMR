#!/bin/bash

hostname
nvidia-smi
task=MCQA

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12358 $task/train-MCQA.py \
        --output_dir ./saved_models/mcqa-PMR-large \
        --model_type roberta \
        --model_name_or_path DAMO-NLP-SG/PMR-large --cache_dir ../cache \
        --data_path MCTest \
        --do_train --do_eval --do_lower_case \
        --learning_rate 1e-5  \
        --num_train_epochs 8 \
        --per_gpu_eval_batch_size=16  \
        --per_gpu_train_batch_size=2 \
        --max_seq_length 512 --max_query_length 128 \
        --save_steps 0 --logging_steps 2000\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training
