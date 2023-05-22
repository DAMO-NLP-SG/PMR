import json
import bz2
from tqdm import tqdm
import logging
import os
import random
import argparse
import nltk
from nltk.corpus import stopwords
from multiprocessing import cpu_count
import torch
import numpy as np
try:
    nltk_stopwords = stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk_stopwords = stopwords.words('english')
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
logging.getLogger().setLevel(logging.INFO)

def read_bz2(addr, return_length=False):
    f = []
    total_length = 0
    with bz2.open(addr) as reader:
        for line in tqdm(reader, desc="reading from {}".format(addr)):
            js_line = json.loads(line)
            total_length += len(js_line[1])
            f.append(js_line)
    f = dict(f)
    if return_length:
        return f, total_length
    else:
        return f

def save_new_file(new_e2c=None, new_e2c_f=None, new_e2p=None, new_e2p_f=None):
    if new_e2c is not None:
        with bz2.BZ2File(new_e2c_f, 'w') as writer:
            for item in tqdm(new_e2c.items(), desc='save file at ' + new_e2c_f):
                output_item = json.dumps(item, ensure_ascii=False) + "\n"
                writer.write(output_item.encode("utf-8"))
    if new_e2p is not None:
        with bz2.BZ2File(new_e2p_f, 'w') as writer:
            for item in tqdm(new_e2p.items(), desc='save file at ' + new_e2p_f):
                output_item = json.dumps(item, ensure_ascii=False) + "\n"
                writer.write(output_item.encode("utf-8"))

def filter_max(args):
    e2p_addr = os.path.join(args.file, 'processed', "e2p.bz2")
    e2c_addr = os.path.join(args.file, 'processed', "e2c.bz2")
    e2c = read_bz2(e2c_addr)
    e2p = read_bz2(e2p_addr)
    logging.info("length of e2c: %d", len(e2c))
    logging.info("length of e2p: %d", len(e2p))
    rm_keys = []
    keep_keys = {}
    keys = list(e2c.keys())
    total_count = sum([len(e2c[x]) for x in e2c])
    p = args.max / total_count
    for key in keys:
        items = e2c[key]
        sampled_items = [item for item in items if random.random() < p]
        if len(sampled_items) != 0:
            e2c[key] = sampled_items
            keep_keys[key] = 1
            for item in sampled_items:
                keep_keys[item[0]] = 1
        else:
            rm_keys.append(key)
            del e2c[key]
    logging.info(
        "Totally we have %d items. We remove entity anchors such that the maximum item number is around %d, some removed examples are %s.",
        total_count, args.max,
        rm_keys[:100])
    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + '{}'.format(args.max) + '.bz2')
    e2p_save_dir = os.path.join(args.file, 'processed', 'e2p_' + '{}'.format(args.max) + '.bz2')
    new_e2p = {}
    for key in keep_keys:
        new_e2p[key] = e2p.pop(key)
    save_new_file(e2c, e2c_save_dir, new_e2p, e2p_save_dir)


def reduce_process(output_queue, all_features):
    '''
    write the processed data to the e2p_and_e2c from output_queue
    :param output_queue:
    :param all_features:
    :return:
    '''
    while True:
        target, doc = output_queue.get()
        if target is not None:
            # logging.info("do_retrieval at {}".format(target))
            all_features[target] = doc
        else:
            logging.info('Quit Reducer')
            break


def filter_full_random(args):

    e2c_addr = os.path.join(args.file, 'processed', "e2c.bz2")
    e2c = read_bz2(e2c_addr)
    e2c_mix = {}
    original_length = {e2c_addr:0}

    for target in e2c:
        context = e2c[target]
        original_length[e2c_addr] += len(context)
        if len(context) < args.fre:
            continue

        new_context = [tuple(x[:4]) for x in context]
        e2c_mix[target] = new_context
    del e2c

    mix_length = 0
    for target in e2c_mix:
        context = e2c_mix[target]
        e2c_mix[target] = random.sample(context, k=min(len(context), args.k))
        mix_length += len(e2c_mix[target])
    for addr in original_length:
        logging.info("The number of query-context in {} is {}.".format(addr, original_length[addr]))
    logging.info("Total number of query-context in e2c file is {}.".format(sum(original_length[addr] for addr in original_length)))
    logging.info("After sampling the query-context in e2c file, we get {} pairs.".format(mix_length))

    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + args.sample_data + '.bz2')
    save_new_file(e2c_mix, e2c_save_dir)

def filter(args):
    if args.method == 'fre':
        filter_full_random(args)
    elif args.method == 'max':
        filter_max(args)



if __name__ == "__main__":
    default_process_count = max(1, cpu_count() - 1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="./en",
                        help="e2c file directory")
    parser.add_argument("--fre", type=int, default=10,
                        help="the word frequency to apply sampling")
    parser.add_argument("--k", type=int, default=10,
                        help="sample number for each entity")
    parser.add_argument("--max", type=int, default=None,
                        help="we sample instances to a max number.")
    parser.add_argument("--def_num", type=int, default=1,
                        help="we use the first def_num sentences as the definition for each entity")
    parser.add_argument("--processes", type=int, default=default_process_count,
                        help="Number of processes to use (default %(default)s)")
    parser.add_argument("--method", type=str, default='bm25', choices=['max', 'fre'],
                        help="perform what kinds of filtering method")
    parser.add_argument("--bm25_threshold", type=float, default=1,
                        help="the rate to sample bm25 ranked data")
    parser.add_argument("--simcse_threshold", type=float, default=1,
                        help="the rate to sample bm25 ranked data")
    parser.add_argument("--mix_threshold", type=float, default=0.6,
                        help="the rate to sample mix ranked data")
    parser.add_argument("--max_context", type=int, default=50,
                        help="Some entities have large size of context. We make some random sampling before further preprocessing. ")
    parser.add_argument("--batch_size", default=3, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--sample_data", type=str, default='bm25-simcse-30', help="the sampled file file for generating data")
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
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--cache_dir",
        default="./cache",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = torch.cuda.device_count()
    args.device = device
    filter(args)
