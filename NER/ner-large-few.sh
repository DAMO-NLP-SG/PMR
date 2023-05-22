#!/bin/bash

hostname
nvidia-smi
task=NER
  for lr in 5e-5 1e-4
  do
for data in conll03 wnut ace2004 ace2005
do
for i in 42_4 43_4 44_4 45_4 46_4 42_8 43_8 44_8 45_8 46_8 42_16 43_16 44_16 45_16 46_16 42_32 43_32 44_32 45_32 46_32 42_64 43_64 44_64 45_64 46_64
do

CUDA_VISIBLE_DEVICES=0 python  $task/train-NER.py \
        --output_dir ./saved_models/$data-PMR-large-few \
        --model_type roberta \
        --model_name_or_path  DAMO-NLP-SG/PMR-large --cache_dir ../cache \
        --data_path fewshot_${data}_$i \
        --do_train --do_eval --do_lower_case \
        --learning_rate $lr \
        --num_train_epochs 20 --max_steps 200 \
        --per_gpu_eval_batch_size=64  \
        --per_gpu_train_batch_size=12 \
        --max_seq_length 192 \
        --save_steps 0 --logging_steps 200\
        --fp16  --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training
done
done
done