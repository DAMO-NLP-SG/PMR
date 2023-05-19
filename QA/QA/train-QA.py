# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team, and The Alibaba Damo Academy team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import argparse
from decimal import Decimal
import glob
import logging
import os
import random
import json
import transformers
from model import RoBERTa_PMR
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import collate_to_max_length_roberta, generate_span
from data import QADataset, cache_QAexamples
from transformers.trainer_utils import is_main_process
import numpy as np
import torch
import datetime
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm, trange
from mrqa_official_eval import read_answers
from mrqa_official_eval import evaluate as evaluateQA



logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def save_result_to_disk(result, path):
    with open(path, "w") as f:
        json.dump(result, f)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_random_subset(dataset, keep_frac=1):
    """
    Takes a random subset of dataset, where a keep_frac fraction is kept.
    """
    keep_indices = [i for i in range(len(dataset)) if np.random.random() < keep_frac]
    subset_dataset = torch.utils.data.Subset(dataset, keep_indices)
    return subset_dataset


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.keep_frac < 1:
        train_dataset = get_random_subset(train_dataset, keep_frac=args.keep_frac)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_to_max_length_roberta, num_workers=4)

    if args.max_steps > 0 and args.max_steps >= len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.max_steps = 0

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_rate * t_total), num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    best_f1 = -1
    best_em = -1
    tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    if args.logging_steps > 0 and t_total // 5 <= args.logging_steps:
        args.logging_steps = t_total // 5
        logger.info("  Change logging steps to = %d", args.logging_steps)
    for i_e, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # with torch.no_grad():
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "label_mask": batch[3],
                "match_labels": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                raise NotImplementedError

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics args.num_train_epochs - 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        if args.data_path != "out_dev":
                            results, _ = evaluate(args, args.data_path, model, tokenizer, prefix=global_step,)
                            if results['f1'] > best_f1:
                                best_f1 = results['f1']
                                best_em = results['exact_match']
                                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format('best'))
                                # Take care of distributed/parallel training
                                model_to_save = model.module if hasattr(model, "module") else model
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)

                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                logger.info("Saving model checkpoint to %s", output_dir)

                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                        else:
                            for data_path in ["BioASQ", "DROP", "DuoRC", "RACE", "RelationExtraction", "TextbookQA"]:
                                results, _ = evaluate(args, data_path, model, tokenizer, prefix=global_step, split="out_dev")
                    logger.warning(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logging_loss = tr_loss
                # Save model checkpoint
                if  args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        logger.warning(
            "Average loss: %s at global step: %s",
            str((tr_loss - epoch_loss) / (step + 1)),
            str(global_step),
        )
        epoch_loss = tr_loss
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step, best_f1, best_em


def evaluate(args, data_path, model, tokenizer, prefix="", split="in_dev"):
    if data_path.startswith("SQuAD"):
        data_path = "SQuAD"
    dataset = load_and_cache_examples(args, tokenizer, evaluate=True, data_path=data_path, split=split)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_to_max_length_roberta, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = {}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                raise NotImplementedError
            outputs = model(**inputs)
        span_logits = outputs['span_logits']
        item2span = generate_span(match_logits=span_logits, label_mask=batch[3], data_itemid=batch[5])
        all_results.update(item2span)

    predictions = {}
    for item in all_results:
        feature = dataset.all_data[item]
        context = feature.context
        context_str = feature.context_str
        qid = feature.qid
        token_to_orig_map = feature.token_to_orig_map
        span_index, logit, _ = all_results[item]
        word_index = (token_to_orig_map[span_index[0]], token_to_orig_map[span_index[1]])
        if word_index[1] + 1 < len(context):
            str_start = context[word_index[0]][1]
            str_end = context[word_index[1] + 1][1]
            span = context_str[str_start:str_end].strip()
        else:
            str_start = context[word_index[0]][1]
            span = context_str[str_start:].strip()

        if qid not in predictions:
            predictions[qid] = (span, logit)
        else:
            span_max, logit_max = predictions[qid]
            if logit > logit_max:
                predictions[qid] = (span, logit)
    predictions = {x: predictions[x][0] for x in predictions}
    preds_file = os.path.join(args.output_dir, args.data_path + '-predictions.json')
    with open(preds_file, 'w') as writer:
        json.dump(predictions, writer)
    if data_path.startswith('fewshot'):
        if split == 'test':
            gold_answers = read_answers(os.path.join('Data', 'mrqa-few-shot', data_path.split('_')[1], "dev.jsonl"))
            scores = evaluateQA(gold_answers, predictions)
            logger.warning(f"EVAL INFO {data_path} -> valid_f1 is: {scores['f1']}; exact match is: {scores['exact_match']}.")
        else:
            gold_answers = read_answers(os.path.join('Data', 'mrqa-few-shot', data_path.split('_')[1], data_path.split('_')[1] + '-train-seed-42-num-examples-512.jsonl'))
            scores = evaluateQA(gold_answers, predictions, skip_no_answer=True)
            logger.warning(f"EVAL INFO {data_path} -> valid_f1 is: {scores['f1']}; exact match is: {scores['exact_match']}.")
    else:
        gold_answers = read_answers(os.path.join("Data", split, data_path + ".jsonl.gz"))
        scores = evaluateQA(gold_answers, predictions)
        logger.warning(f"EVAL INFO {data_path} -> valid_f1 is: {scores['f1']}; exact match is: {scores['exact_match']}.")
    return scores, predictions


def load_and_cache_examples(args, tokenizer, evaluate=False, data_path=None, split="train"):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    if data_path is None:
        data_path = args.data_path
    if data_path.startswith('fewshot') and split == 'test':
        cached_features_file = os.path.join(
            './Data',
            "cached_{}_{}_{}_{}_{}".format(
                data_path.split("_")[0] + "_" + data_path.split("_")[1],
                split,
                list(filter(None, args.model_type.split("/"))).pop(),
                str(args.max_query_length),
                str(args.max_seq_length),
            ),
        )
    elif data_path.startswith('fewshot') and split == 'in_dev':
        cached_features_file = os.path.join(
            './Data',
            "cached_{}_{}_{}_{}_{}".format(
                data_path.split("_")[0] + "_" + data_path.split("_")[1] + "_" + data_path.split("_")[3],
                split,
                list(filter(None, args.model_type.split("/"))).pop(),
                str(args.max_query_length),
                str(args.max_seq_length),
            ),
        )
    else:
        cached_features_file = os.path.join(
            './Data',
            "cached_{}_{}_{}_{}_{}".format(
                data_path,
                split,
                list(filter(None, args.model_type.split("/"))).pop(),
                str(args.max_query_length),
                str(args.max_seq_length),
            ),
        )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    elif split == "train" and data_path == 'out_dev':
        features = cache_QAexamples(args, tokenizer, ["SQuAD", "NewsQA", "TriviaQA", "SearchQA", "HotpotQA", "NaturalQuestions"], split,  sample_num=75000)
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.info(" data saved at {}.".format(cached_features_file))
    elif data_path.startswith('fewshot'):
        features = cache_QAexamples(args, tokenizer, data_path, split)
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.info(" data saved at {}.".format(cached_features_file))
    else:
        features = cache_QAexamples(args, tokenizer, data_path, split)

        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.info(" data saved at {}.".format(cached_features_file))

    logger.info("load data from {}.".format(data_path))
    dataset = QADataset(features=features,
                        data_name=data_path,
                        tokenizer=tokenizer,
                        max_length=args.max_seq_length,
                        max_query_length=args.max_query_length,
                        pad_to_maxlen=False,
                        is_training=not evaluate,
                        )
    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    return dataset



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--data_path",
        # choices=["out_dev", "SQuAD", "NewsQA", "TriviaQA", 'SearchQA', 'HotpotQA', "NaturalQuestions"],
        default="SQuAD",
        type=str,
        help="The input file.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_rate", default=0.06, type=float, help="Linear warmup over warmup_rate.")
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--keep_frac", type=float, default=1.0, help="The fraction of the balanced dataset to keep.")
    parser.add_argument(
        "--drop",
        default=0.1,
        type=float,
        help="dropout rate",
    )
    parser.add_argument(
        "--projection_intermediate_hidden_size",
        default=1024,
        type=int,
        help="span classifier intermediate hidden size",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=18000))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,
        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    config.projection_intermediate_hidden_size = args.projection_intermediate_hidden_size
    config.hidden_dropout_prob = args.drop
    model = RoBERTa_PMR.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        # NOTE: balances dataset in load_and_cache_examples
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        # load_and_cache_examples(args, tokenizer, evaluate=True, split='in_dev')
        # load_and_cache_examples(args, tokenizer, evaluate=True, split='out_dev')
        global_step, tr_loss, best_f1, best_em = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            if args.data_path != "out_dev":
                results, predictions = evaluate(args, args.data_path, model, tokenizer, prefix='')
                preds_file = os.path.join(args.output_dir, args.data_path + '-predictions.json')
                with open(preds_file, 'w') as writer:
                    json.dump(predictions, writer)
            else:
                for data_path in ["BioASQ", "DROP", "DuoRC", "RACE", "RelationExtraction", "TextbookQA"]:
                    results, predictions = evaluate(args, data_path, model, tokenizer, prefix='', split="out_dev")
                    preds_file = os.path.join(args.output_dir, args.data_path + '-predictions.json')
                    with open(preds_file, 'w') as writer:
                        json.dump(predictions, writer)
            if results['f1'] > best_f1:
                best_f1 = results['f1']
                best_em = results['exact_match']
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format('best'))
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
            else:
                model = RoBERTa_PMR.from_pretrained(
                    os.path.join(args.output_dir, 'checkpoint-best'),
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                )
                model.to(args.device)
            if args.data_path.startswith('fewshot'):
                test_results, test_predictions = evaluate(args, args.data_path, model, tokenizer, prefix='', split='test')
                test_f1, test_em = test_results['f1'], test_results['exact_match']
                test_f1, test_em = Decimal(test_f1).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP"), Decimal(
                    test_em).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP")
            else:
                test_f1, test_em = None, None
            best_f1, best_em = Decimal(best_f1).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP"), Decimal(best_em).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP")
            logger.warning(f"EVAL dev at {args.data_path} -> f1 is: {best_f1}. em is: {best_em}.")
            logger.warning(f"EVAL test at {args.data_path} -> f1 is: {test_f1}. em is: {test_em}.")
            if "_" in args.data_path:
                data_name = args.data_path.split("_")[1].lower()
            else:
                data_name = args.data_path.split('/')[-1].lower()
            with open('result_' + data_name + '.txt', 'a') as writer:
                save_string = "\t".join([args.model_name_or_path, args.data_path, str(args.learning_rate), str(args.num_train_epochs), str(args.per_gpu_train_batch_size), "/".join([str(best_f1), str(best_em)]), "/".join([str(test_f1), str(test_em)])])
                writer.write(save_string + '\n')
        else:
            global_step = ""
            if args.data_path != "out_dev":
                results, predictions = evaluate(args, args.data_path, model, tokenizer, prefix='')
                preds_file = os.path.join(args.output_dir, args.data_path + '-predictions.json')
                with open(preds_file, 'w') as writer:
                    json.dump(predictions, writer)

            else:
                for data_path in ["BioASQ", "DROP", "DuoRC", "RACE", "RelationExtraction", "TextbookQA"]:
                    results, predictions = evaluate(args, data_path, model, tokenizer, prefix='', split="out_dev")
                    preds_file = os.path.join(args.output_dir, args.data_path + '-predictions.json')
                    with open(preds_file, 'w') as writer:
                        json.dump(predictions, writer)

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )

if __name__ == "__main__":
    main()
