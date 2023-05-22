#!/bin/bash

hostname
nvidia-smi
task=mNER


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 $task/train-NER.py \
        --output_dir ./saved_models/conll-mPMR-large \
        --model_type xlmr \
        --model_name_or_path DAMO-NLP-SG/mPMR-large --cache_dir ../cache \
        --data_path conll \
        --do_train --do_eval --do_lower_case \
        --learning_rate 1e-5 \
        --num_train_epochs 10 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=16 \
        --max_seq_length 192 --max_query_length 32\
        --save_steps 0 --logging_steps 4000\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training
