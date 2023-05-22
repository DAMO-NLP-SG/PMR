#!/bin/bash

hostname
nvidia-smi
task=mABSA

for seed in 42 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=0 python $task/train-ABSA.py \
        --output_dir ./saved_models/rest-mPMR-base \
        --model_type xlmr \
        --model_name_or_path DAMO-NLP-SG/mPMR-base --cache_dir ../cache \
        --data_path rest \
        --do_train --do_eval --do_lower_case \
        --learning_rate 2e-5 \
        --num_train_epochs 20 \
        --per_gpu_eval_batch_size=64  \
        --per_gpu_train_batch_size=32 \
        --max_seq_length 192 --max_query_length 32 --seed $seed\
        --save_steps 0 --logging_steps 400\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training
done