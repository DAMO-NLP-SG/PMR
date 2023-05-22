
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 mPMR/train.py \
        --output_dir ./saved_models/mPMR_xlmr-large-full-random10_negative-epoch3-lr1e-5-mix24 \
        --model_type xlmr \
        --model_name_or_path xlm-roberta-large --cache_dir ./cache \
        --data_path ./mix24/processed \
        --do_train --do_eval --do_lower_case \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --per_gpu_eval_batch_size=40  \
        --per_gpu_train_batch_size=20 \
        --max_seq_length 512 --max_query_length 128 --sample_data 'full-random-10_negative' \
        --save_steps 50000 --logging_steps 50000\
        --fp16 --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training --lazy_load