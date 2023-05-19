import copy
import json
import bz2
from tqdm import tqdm
import argparse
import logging
import os
import random
import argparse
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import string
from multiprocessing import Queue, Process, cpu_count, Manager
import time
import torch
import math
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
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

def filter_fre(args):
    e2p_addr = os.path.join(args.file, 'processed', "e2p.bz2")
    e2c_addr = os.path.join(args.file, 'processed', "e2c.bz2")
    e2c = read_bz2(e2c_addr)
    e2p = read_bz2(e2p_addr)
    logging.info("length of e2c: %d", len(e2c))
    logging.info("length of e2p: %d", len(e2p))
    rm_keys = []
    keep_keys = {}
    keys = list(e2c.keys())

    for key in keys:
        if len(e2c[key]) < args.fre:
            if len(e2c[key]) == args.fre - 1:
                rm_keys.append(key)
            del e2c[key]
        else:
            sampled_items = random.sample(e2c[key], k=args.k)
            e2c[key] = sampled_items
            keep_keys[key] = 1
            for item in sampled_items:
                keep_keys[item[0]] = 1
    logging.info("We remove entity anchors with frequency lower than %d, some examples are %s.", args.fre,
                 rm_keys[:100])
    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + '{}-{}'.format(args.fre, args.k) + '.bz2')
    e2p_save_dir = os.path.join(args.file, 'processed', 'e2p_' + '{}-{}'.format(args.fre, args.k) + '.bz2')
    new_e2p = {}
    for key in keep_keys:
        new_e2p[key] = e2p.pop(key)
    save_new_file(e2c, e2c_save_dir, new_e2p, e2p_save_dir)

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

def prepare_bm25_corpus(args, len_data, jobs_queue, output_queue, time_start):
    while True:
        i_t, target, documents = jobs_queue.get()
        if target is not None:
            if i_t % 100000 == 0:
                passed_time = time.time() - time_start
                estimate_time = passed_time / (i_t + 1) * (len_data - i_t)
                logging.info("prepare_bm25_corpus at No. %d entity '%s', time cost is %f. Estimated remaining time is %f", i_t, target, passed_time, estimate_time)
            documents_clean = []
            def_num = args.def_num
            definition = documents[: def_num]
            definition = [word for sen in definition for word in sen]  # linearize
            offset = 0
            if len(definition) < 30:
                # wrongly segment the title caused by sentence segmenter error.
                # We add more sentences as definition
                for offset in range(1, 10):
                    definition = documents[: def_num + offset]
                    definition = [word for sen in definition for word in sen]
                    if len(definition) >= 30:
                        break
            context_offset = def_num + offset

            for context_id in range(len(documents)):
                context_sentence = documents[context_id]
                context_sentence = [x.lower() for x in context_sentence if
                                    x not in nltk_stopwords and x not in string.punctuation]
                documents_clean.append(context_sentence)
            e2p_new = {
                "offset": context_offset,
                # 'documents': documents,
                'definition_clean': [word for sen in documents_clean[:context_offset] for word in sen],
                "documents_clean": documents_clean,
            }
            output_queue.put((target, e2p_new))
        else:
            logging.info('Quit worker')
            break

def do_retrieval(args, len_data, bm25, jobs_queue, output_queue, time_start):
    while True:
        i_t, batch = jobs_queue.get()
        if i_t is not None:
            if i_t % 1000 == 0:
                passed_time = time.time() - time_start
                estimate_time = passed_time / (i_t + 1) * (len_data - i_t)
                logging.info("do_retrieval at No. %d batch, time cost is %f. Estimated remaining time is %f", i_t, passed_time, estimate_time)

            bm25_scores = bm25.get_batch_scores(args, batch)
            output_queue.put((i_t, bm25_scores))
        else:
            logging.info('Quit worker')
            break

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

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
# Tokenize input texts


def norm(a, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


def prepare_simcse_corpus(args, len_data, jobs_queue, output_queue, time_start):
    while True:
        i_t, target, documents = jobs_queue.get()
        if target is not None:
            if i_t % 100000 == 0:
                passed_time = time.time() - time_start
                estimate_time = passed_time / (i_t + 1) * (len_data - i_t)
                logging.info("prepare_bm25_corpus at No. %d entity '%s', time cost is %f. Estimated remaining time is %f", i_t, target, passed_time, estimate_time)
            def_num = args.def_num
            definition = documents[: def_num]
            definition = [word for sen in definition for word in sen]  # linearize
            offset = 0
            if len(definition) < 30:
                # wrongly segment the title caused by sentence segmenter error.
                # We add more sentences as definition
                for offset in range(1, 10):
                    definition = documents[: def_num + offset]
                    definition = [word for sen in definition for word in sen]
                    if len(definition) >= 30:
                        break
            context_offset = def_num + offset
            e2p_new = {
                'definition': " ".join(definition),
                "documents": [" ".join(x) for x in documents]
            }
            output_queue.put((target, e2p_new))
        else:
            logging.info('Quit worker')
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

def filter_mix_30(args):

    e2c_bm25_addr = os.path.join(args.file, 'processed', "e2c_bm25_50.bz2")
    e2c_simcse_addr = [os.path.join(args.file, 'processed', 'e2c_' + 'simcse_50.' + str(i) + '.bz2') for i in range(args.n_gpu)]
    e2c_bm25 = read_bz2(e2c_bm25_addr)
    e2c_mix = {}
    original_length = {e2c_bm25_addr:0}

    for target in e2c_bm25:
        context = e2c_bm25[target]
        original_length[e2c_bm25_addr] += len(context)
        if len(context) < 3: # the length of 50% of the context < 3 is equivalent to the length of 100% of the context < 5
            continue
        if args.mix_threshold != 1:
            keep_index = math.ceil(len(context) * args.mix_threshold)
            context = context[:keep_index]
        new_context = [tuple(x[:4]) for x in context]
        e2c_mix[target] = new_context
    del e2c_bm25

    for e2c_simcse_addr_one in e2c_simcse_addr:
        original_length[e2c_simcse_addr_one] = 0
        e2c_simcse_one = read_bz2(e2c_simcse_addr_one)
        for target in e2c_simcse_one:
            context = e2c_simcse_one[target]
            original_length[e2c_simcse_addr_one] += len(context)
            if len(context) < 3:  # the length of 50% of the context < 3 is equivalent to the length of 100% of the context < 5
                continue
            if args.mix_threshold != 1:
                keep_index = math.ceil(len(context) * args.mix_threshold)
                context = context[:keep_index]
            new_context = [tuple(x[:4]) for x in context]
            if target in e2c_mix:
                e2c_mix[target] = list(set(e2c_mix[target] + new_context))
            else:
                e2c_mix[target] = new_context
        del e2c_simcse_one
    mix_length = 0
    for target in e2c_mix:
        mix_length += len(e2c_mix[target])
    for addr in original_length:
        logging.info("The number of query-context in {} is {}.".format(addr, original_length[addr]))
    logging.info("Total number of query-context in all bm25 and simcse filtering result is {}.".format(sum(original_length[addr] for addr in original_length)))
    logging.info("After mixing the query-context in all bm25 and simcse file, we get {} pairs.".format(mix_length))

    keep_keys = {}
    e2c_mix_keys = list(e2c_mix.keys())
    for key in e2c_mix_keys:
        context = e2c_mix[key]
        keep_keys[key] = 1
        for x in context:
            keep_keys[x[0]] = 1
    e2p_addr = os.path.join(args.file, 'processed', "e2p.bz2")
    e2p = read_bz2(e2p_addr)
    e2p_mix = {}
    for key in keep_keys:
        e2p_mix[key] = e2p.pop(key)
    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + args.sample_data + '.bz2')
    e2p_save_dir = os.path.join(args.file, 'processed', 'e2p_' + args.sample_data + '.bz2')
    save_new_file(e2c_mix, e2c_save_dir, e2p_mix, e2p_save_dir)

def filter_mix_50_random_low5(args):

    e2c_bm25_addr = os.path.join(args.file, 'processed', "e2c_bm25_50.bz2")
    e2c_simcse_addr = [os.path.join(args.file, 'processed', 'e2c_' + 'simcse_50.' + str(i) + '.bz2') for i in range(args.n_gpu)]
    e2c_bm25 = read_bz2(e2c_bm25_addr)
    e2c_mix = {}
    original_length = {e2c_bm25_addr:0}

    for target in e2c_bm25:
        context = e2c_bm25[target]
        original_length[e2c_bm25_addr] += len(context)
        if len(context) < 3: # the length of 50% of the context < 3 is equivalent to the length of 100% of the context < 5
            continue

        new_context = [tuple(x[:4]) for x in context]
        e2c_mix[target] = new_context
    del e2c_bm25

    for e2c_simcse_addr_one in e2c_simcse_addr:
        original_length[e2c_simcse_addr_one] = 0
        e2c_simcse_one = read_bz2(e2c_simcse_addr_one)
        for target in e2c_simcse_one:
            context = e2c_simcse_one[target]
            original_length[e2c_simcse_addr_one] += len(context)
            if len(context) < 3:  # the length of 50% of the context < 3 is equivalent to the length of 100% of the context < 5
                continue
            new_context = [tuple(x[:4]) for x in context]
            if target in e2c_mix:
                e2c_mix[target] = list(set(e2c_mix[target] + new_context))
            else:
                e2c_mix[target] = new_context
        del e2c_simcse_one
    mix_length = 0
    for target in e2c_mix:
        context = e2c_mix[target]
        e2c_mix[target] = random.sample(context, k=min(len(context), args.k))
        mix_length += len(e2c_mix[target])
    for addr in original_length:
        logging.info("The number of query-context in {} is {}.".format(addr, original_length[addr]))
    logging.info("Total number of query-context in all bm25 and simcse filtering result is {}.".format(sum(original_length[addr] for addr in original_length)))
    logging.info("After mixing the query-context in all bm25 and simcse file, we get {} pairs.".format(mix_length))

    keep_keys = {}
    e2c_mix_keys = list(e2c_mix.keys())
    for key in e2c_mix_keys:
        context = e2c_mix[key]
        keep_keys[key] = 1
        for x in context:
            keep_keys[x[0]] = 1
    e2p_addr = os.path.join(args.file, 'processed', "e2p.bz2")
    e2p = read_bz2(e2p_addr)
    e2p_mix = {}
    for key in keep_keys:
        e2p_mix[key] = e2p.pop(key)
    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + args.sample_data + '.bz2')
    e2p_save_dir = os.path.join(args.file, 'processed', 'e2p_' + args.sample_data + '.bz2')
    save_new_file(e2c_mix, e2c_save_dir, e2p_mix, e2p_save_dir)

def filter_mix_50_random(args):

    e2c_bm25_addr = os.path.join(args.file, 'processed', "e2c_bm25_50.bz2")
    e2c_simcse_addr = [os.path.join(args.file, 'processed', 'e2c_' + 'simcse_50.' + str(i) + '.bz2') for i in range(args.n_gpu)]
    e2c_bm25 = read_bz2(e2c_bm25_addr)
    e2c_mix = {}
    original_length = {e2c_bm25_addr:0}

    for target in e2c_bm25:
        context = e2c_bm25[target]
        original_length[e2c_bm25_addr] += len(context)
        if len(context) < 5: # the length of 50% of the context < 5 is equivalent to the length of 100% of the context < 10
            continue

        new_context = [tuple(x[:4]) for x in context]
        e2c_mix[target] = new_context
    del e2c_bm25

    for e2c_simcse_addr_one in e2c_simcse_addr:
        original_length[e2c_simcse_addr_one] = 0
        e2c_simcse_one = read_bz2(e2c_simcse_addr_one)
        for target in e2c_simcse_one:
            context = e2c_simcse_one[target]
            original_length[e2c_simcse_addr_one] += len(context)
            if len(context) < 5:  # the length of 50% of the context < 5 is equivalent to the length of 100% of the context < 10
                continue
            new_context = [tuple(x[:4]) for x in context]
            if target in e2c_mix:
                e2c_mix[target] = list(set(e2c_mix[target] + new_context))
            else:
                e2c_mix[target] = new_context
        del e2c_simcse_one
    mix_length = 0
    for target in e2c_mix:
        context = e2c_mix[target]
        e2c_mix[target] = random.sample(context, k=min(len(context), args.k))
        mix_length += len(e2c_mix[target])
    for addr in original_length:
        logging.info("The number of query-context in {} is {}.".format(addr, original_length[addr]))
    logging.info("Total number of query-context in all bm25 and simcse filtering result is {}.".format(sum(original_length[addr] for addr in original_length)))
    logging.info("After mixing the query-context in all bm25 and simcse file, we get {} pairs.".format(mix_length))

    keep_keys = {}
    e2c_mix_keys = list(e2c_mix.keys())
    for key in e2c_mix_keys:
        context = e2c_mix[key]
        keep_keys[key] = 1
        for x in context:
            keep_keys[x[0]] = 1
    e2p_addr = os.path.join(args.file, 'processed', "e2p.bz2")
    e2p = read_bz2(e2p_addr)
    e2p_mix = {}
    for key in keep_keys:
        e2p_mix[key] = e2p.pop(key)
    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + args.sample_data + '.bz2')
    e2p_save_dir = os.path.join(args.file, 'processed', 'e2p_' + args.sample_data + '.bz2')
    save_new_file(e2c_mix, e2c_save_dir, e2p_mix, e2p_save_dir)

def filter_mix_50_bottom10(args):

    e2c_bm25_addr = os.path.join(args.file, 'processed', "e2c_bm25_50.bz2")
    e2c_simcse_addr = [os.path.join(args.file, 'processed', 'e2c_' + 'simcse_50.' + str(i) + '.bz2') for i in range(args.n_gpu)]
    e2c_bm25 = read_bz2(e2c_bm25_addr)
    e2c_mix = {}
    original_length = {e2c_bm25_addr:0}

    for target in e2c_bm25:
        context = e2c_bm25[target]
        original_length[e2c_bm25_addr] += len(context)
        if len(context) < args.k // 2:
            continue
        context = sorted(context, key=lambda x: x[4])[:args.k // 2]
        new_context = [tuple(x[:4]) for x in context]
        e2c_mix[target] = new_context
    del e2c_bm25

    for e2c_simcse_addr_one in e2c_simcse_addr:
        original_length[e2c_simcse_addr_one] = 0
        e2c_simcse_one = read_bz2(e2c_simcse_addr_one)
        for target in e2c_simcse_one:
            context = e2c_simcse_one[target]
            original_length[e2c_simcse_addr_one] += len(context)
            if len(context) < args.k // 2:
                continue
            context = sorted(context, key=lambda x: x[4])
            new_context = [tuple(x[:4]) for x in context]
            if target in e2c_mix:
                all_context = e2c_mix[target]
                bm25_length = len(all_context)
                assert bm25_length < args.k
                simcse_length = 0
                for i in range(len(new_context)):
                    if bm25_length + simcse_length == args.k:
                        break
                    else:
                        if new_context[i] not in all_context:
                            all_context.append(new_context[i])
                            simcse_length += 1

                e2c_mix[target] = all_context
            else:
                e2c_mix[target] = new_context[:args.k]
        del e2c_simcse_one
    mix_length = 0
    for target in e2c_mix:
        mix_length += len(e2c_mix[target])
    for addr in original_length:
        logging.info("The number of query-context in {} is {}.".format(addr, original_length[addr]))
    logging.info("Total number of query-context in all bm25 and simcse filtering result is {}.".format(sum(original_length[addr] for addr in original_length)))
    logging.info("After mixing the query-context in all bm25 and simcse file, we get {} pairs.".format(mix_length))

    keep_keys = {}
    e2c_mix_keys = list(e2c_mix.keys())
    for key in e2c_mix_keys:
        context = e2c_mix[key]
        keep_keys[key] = 1
        for x in context:
            keep_keys[x[0]] = 1
    e2p_addr = os.path.join(args.file, 'processed', "e2p.bz2")
    e2p = read_bz2(e2p_addr)
    e2p_mix = {}
    for key in keep_keys:
        e2p_mix[key] = e2p.pop(key)
    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + args.sample_data + '.bz2')
    e2p_save_dir = os.path.join(args.file, 'processed', 'e2p_' + args.sample_data + '.bz2')
    save_new_file(e2c_mix, e2c_save_dir, e2p_mix, e2p_save_dir)

def filter_mix_50_top10(args):

    e2c_bm25_addr = os.path.join(args.file, 'processed', "e2c_bm25_50.bz2")
    e2c_simcse_addr = [os.path.join(args.file, 'processed', 'e2c_' + 'simcse_50.' + str(i) + '.bz2') for i in range(args.n_gpu)]
    e2c_bm25 = read_bz2(e2c_bm25_addr)
    e2c_mix = {}
    original_length = {e2c_bm25_addr:0}

    for target in e2c_bm25:
        context = e2c_bm25[target]
        original_length[e2c_bm25_addr] += len(context)
        if len(context) < args.k // 2:
            continue
        context = sorted(context, key=lambda x: -x[4])[:args.k // 2]
        new_context = [tuple(x[:4]) for x in context]
        e2c_mix[target] = new_context
    del e2c_bm25

    for e2c_simcse_addr_one in e2c_simcse_addr:
        original_length[e2c_simcse_addr_one] = 0
        e2c_simcse_one = read_bz2(e2c_simcse_addr_one)
        for target in e2c_simcse_one:
            context = e2c_simcse_one[target]
            original_length[e2c_simcse_addr_one] += len(context)
            if len(context) < args.k // 2:
                continue
            context = sorted(context, key=lambda x: -x[4])
            new_context = [tuple(x[:4]) for x in context]
            if target in e2c_mix:
                all_context = e2c_mix[target]
                bm25_length = len(all_context)
                assert bm25_length < args.k
                simcse_length = 0
                for i in range(len(new_context)):
                    if bm25_length + simcse_length == args.k:
                        break
                    else:
                        if new_context[i] not in all_context:
                            all_context.append(new_context[i])
                            simcse_length += 1

                e2c_mix[target] = all_context
            else:
                e2c_mix[target] = new_context[:args.k]
        del e2c_simcse_one
    mix_length = 0
    for target in e2c_mix:
        mix_length += len(e2c_mix[target])
    for addr in original_length:
        logging.info("The number of query-context in {} is {}.".format(addr, original_length[addr]))
    logging.info("Total number of query-context in all bm25 and simcse filtering result is {}.".format(sum(original_length[addr] for addr in original_length)))
    logging.info("After mixing the query-context in all bm25 and simcse file, we get {} pairs.".format(mix_length))

    keep_keys = {}
    e2c_mix_keys = list(e2c_mix.keys())
    for key in e2c_mix_keys:
        context = e2c_mix[key]
        keep_keys[key] = 1
        for x in context:
            keep_keys[x[0]] = 1
    e2p_addr = os.path.join(args.file, 'processed', "e2p.bz2")
    e2p = read_bz2(e2p_addr)
    e2p_mix = {}
    for key in keep_keys:
        e2p_mix[key] = e2p.pop(key)
    e2c_save_dir = os.path.join(args.file, 'processed', 'e2c_' + args.sample_data + '.bz2')
    e2p_save_dir = os.path.join(args.file, 'processed', 'e2p_' + args.sample_data + '.bz2')
    save_new_file(e2c_mix, e2c_save_dir, e2p_mix, e2p_save_dir)

def filter(args):
    if args.method == 'fre':
        filter_full_random(args)
    elif args.method == 'max':
        filter_max(args)
    elif args.method == "simcse":
        filter_simcse(args)
    elif args.method == "mix":
        filter_mix_50_random_low5(args)
        # filter_mix_50_random(args)
        # filter_mix_50_bottom10(args)
        # filter_mix_50_top10(args)
    elif args.method == "cluster":
        cluster_simcse(args)
    elif args.method == 'merge':
        merge_cluster(args)

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
    parser.add_argument("--method", type=str, default='bm25', choices=['max', 'fre', 'bm25','simcse', 'mix', "cluster", 'merge'],
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
