#!/bin/bash

hostname
nvidia-smi
task=mQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12357 $task/train-QA.py \
        --output_dir ./saved_models/xquad_mlqa-mPMR-large \
        --model_type xlmr \
        --model_name_or_path DAMO-NLP-SG/mPMR-large --cache_dir ../cache \
        --data_path SQuAD \
        --do_train --do_eval --do_lower_case \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=8 \
        --max_seq_length 384 --max_query_length 64\
        --save_steps 0 --logging_steps 2000\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training  # NewsQA HotpotQA NaturalQuestions TriviaQA  SearchQA
