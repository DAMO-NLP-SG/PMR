
import json
import bz2
from tqdm import tqdm
import logging
import os
import random
import argparse

import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

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

def filter_full_random(args):

    e2c_addr = os.path.join(args.file, 'processed', "e2c.bz2")
    e2c = read_bz2(e2c_addr)
    e2c_mix = {}
    original_length = {e2c_addr:0}

    for target in e2c:
        context = e2c[target]
        original_length[e2c_addr] += len(context)
        if len(context) < args.bottom:
            continue

        new_context = [tuple(x) for x in context]
        e2c_mix[target] = new_context
    del e2c

    mix_length = 0
    for target in e2c_mix:
        context = e2c_mix[target]
        e2c_mix[target] = random.sample(context, k=min(len(context), args.up))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="./de",
                        help="e2c file directory")
    parser.add_argument("--bottom", type=int, default=5,
                        help="the word frequency to apply sampling")
    parser.add_argument("--up", type=int, default=5,
                        help="sample number for each entity")
    parser.add_argument("--method", type=str, default='fre', choices=['fre',],
                        help="perform what kinds of filtering method")
    parser.add_argument("--sample_data", type=str, default='full-random-10', help="the sampled file file for generating data")

    args = parser.parse_args()
    filter(args)
