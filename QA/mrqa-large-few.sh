#!/bin/bash

hostname
nvidia-smi
task=QA
  for lr in 5e-5 1e-4
  do
for data in squad hotpotqa naturalquestions textbookqa bioasq triviaqa searchqa newsqa
do
for i in 42_16 43_16 44_16 45_16 46_16 42_32 43_32 44_32 45_32 46_32 42_64 43_64 44_64 45_64 46_64 42_128 43_128 44_128 45_128 46_128 42_1024 43_1024 44_1024 45_1024 46_1024
do
CUDA_VISIBLE_DEVICES=0 python $task/train-QA.py \
        --output_dir ./saved_models/$data-PMR-large-few \
        --model_type roberta \
        --model_name_or_path ../saved_models/PMR-large --cache_dir ../cache \
        --data_path fewshot_${data}_$i \
        --do_train --do_eval --do_lower_case \
        --learning_rate 5e-5 \
        --num_train_epochs 12 --max_steps 200 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=12 \
        --max_seq_length 384 --max_query_length 64\
        --save_steps 0 --logging_steps 1000 \
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training #
done
done
done