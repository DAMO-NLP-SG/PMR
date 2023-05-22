#!/bin/bash

hostname
nvidia-smi
task=NER

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12356 $task/train-NER.py \
        --output_dir ./saved_models/ace05-PMR-base \
        --model_type roberta \
        --model_name_or_path  DAMO-NLP-SG/PMR-base --cache_dir ../cache \
        --data_path ./Data/ace2005 \
        --do_train --do_eval --do_lower_case \
        --learning_rate 2e-5 \
        --num_train_epochs 5\
        --per_gpu_eval_batch_size=64  \
        --per_gpu_train_batch_size=32 \
        --max_seq_length 192 --max_query_length 64\
        --save_steps 0 --logging_steps 400\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training