import json
import bz2
from tqdm import tqdm
import argparse
import random
import logging
import math
import os
from transformers.file_utils import is_torch_available
from transformers.tokenization_utils_base import TruncationStrategy
from multiprocessing import Queue, Process, cpu_count, Manager
import Levenshtein
import time
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    BertConfig,
)
if is_torch_available():
    import torch
random.seed(42)
logging.getLogger().setLevel(logging.INFO)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}
config_PMR = {
    'bert': BertConfig,
    'roberta': AutoConfig,
    'albert': AutoConfig,
}
tokenizer_PMR = {
    'bert': BertTokenizer,
    'roberta': AutoTokenizer,
    'albert': AutoTokenizer,
}
placeholder = "IAmAPalacehodler"

class MRCFeatures:
    '''
    MRC features
    '''
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions,
        end_positions,
        doc_offset,
        len_query,
        len_context,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.doc_offset = doc_offset
        self.len_query = len_query
        self.len_context = len_context

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)

def read_bz2(addr):
    f = []
    with bz2.open(addr) as reader:
        for line in tqdm(reader, desc="reading from {}".format(addr)):
            js_line = json.loads(line)
            f.append(js_line)
    f = dict(f)
    return f

def add_negative(e2c_all, e2c):
    e2c_all_keys = list(e2c_all.keys())
    for target in tqdm(e2c, desc='add negative'):
        context = e2c[target]
        doc_contain_target = [x[0] for x in e2c_all[target]]
        random_keys = random.sample(e2c_all_keys, k=50)
        context_negative = []
        for random_target in random_keys:
            if random_target == target or e2c_all[random_target] == []:
                continue
            possible_context_negative = e2c_all[random_target]
            trial = 0
            while trial < 5:
                trial += 1
                negative_one = random.choice(possible_context_negative)
                if negative_one[0] not in doc_contain_target:
                    negative_one = [negative_one[0], negative_one[1], None, None]
                    context_negative.append(negative_one)
                    break
            if len(context_negative) == len(context):
                break
        assert len(context_negative) == len(context)
        e2c[target] = context + context_negative
    return e2c

def create_mrc(args, e2p, len_data, tokenizer, jobs_queue, output_queue, time_start):
    '''
    Use e2p and e2c to build MRC instances
    :param args:
    :param e2p:
    :param len_data:
    :param tokenizer:
    :param jobs_queue:
    :param output_queue:
    :param time_start:
    :return:
    '''
    while True:
        i_t, item = jobs_queue.get()
        if item is not None:
            target_entity, ref_entity, passage_id, start, end = item
            if i_t % 100000 == 0:
                passed_time = time.time() - time_start
                estimate_time = passed_time / (i_t + 1) * (len_data - i_t)
                logging.info("processing at No. %d entity '%s', time cost is %f. Estimated remaining time is %f", i_t, target_entity, passed_time, estimate_time)
            window, def_num, max_query_length, max_seq_length = args.window, args.def_num, args.max_query_length, args.max_seq_length
            definition = e2p[target_entity][: def_num]
            definition = [word for sen in definition for word in sen]  # linearize
            if len(definition) < 30:
                # wrongly segment the title caused by sentence segmenter error.
                # We add more sentences as definition
                for i in range(1, 10):
                    definition = e2p[target_entity][: def_num + i]
                    definition = [word for sen in definition for word in sen]
                    if len(definition) >= 30:
                        break
            # Mask the answer in the definition: (1) if token is a substring of the title and if token share the Levenshtein similarity with title > 0.5
            definition = " ".join(definition).replace("The " + target_entity, placeholder).replace("the " + target_entity, placeholder).replace(
                "The " + target_entity.lower(), placeholder).replace("the " + target_entity.lower(), placeholder).replace(
                target_entity, placeholder).replace(target_entity.lower(), placeholder).split(" ")
            definition = [placeholder if x in target_entity and (
                        Levenshtein.distance(target_entity, x) / len(target_entity)) < 0.5 else x for x in definition]
            i = 0
            while i < len(definition) - 1:
                if definition[i] == placeholder and definition[i + 1] == placeholder:
                    del definition[i]
                elif definition[i] == placeholder and definition[i + 1] != placeholder:
                    definition[i] = 'it'
                    i += 1
                else:
                    i += 1
            definition[len(definition) - 1] = "it" if definition[len(definition) - 1] == placeholder else definition[
                len(definition) - 1]
            definition = " ".join(definition)
            # query = 'Highlight the parts (if any) related to "{}". Details: {}'.format("it", definition)
            query = definition
            # we consider the passage itself and its former and later window passages as the final context
            context = e2p[ref_entity][max(0, passage_id - window): passage_id + window + 1]
            context = [word for sen in context for word in sen]  # linearize
            if start is not None and end is not None:
                assert (start is not None) and (end is not None)
                target_anchor = e2p[ref_entity][passage_id][start: end + 1]
                span_list = [(i, i + end - start) for i in range(len(context)) if
                             context[i: i + end - start + 1] == target_anchor]
                start_positions = [x[0] for x in span_list]
                end_positions = [x[1] for x in span_list]
            else:
                start_positions = []
                end_positions = []
            # statis
            len_query = len(definition)
            len_context = len(context)

            # convert token to index
            tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
            sequence_added_tokens = (
                tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
                if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
                else tokenizer.model_max_length - tokenizer.max_len_single_sentence
            )

            truncated_query = tokenizer.encode(
                query, add_special_tokens=False, truncation=True, max_length=max_query_length
            )

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(context):
                orig_to_tok_index.append(len(all_doc_tokens))
                if tokenizer.__class__.__name__ in [
                    "RobertaTokenizer",
                    "LongformerTokenizer",
                    "BartTokenizer",
                    "RobertaTokenizerFast",
                    "LongformerTokenizerFast",
                    "BartTokenizerFast",
                ]:
                    sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
                elif tokenizer.__class__.__name__ in [
                    'BertTokenizer'
                ]:
                    sub_tokens = tokenizer.tokenize(token)
                elif tokenizer.__class__.__name__ in [
                    'BertWordPieceTokenizer'
                ]:
                    sub_tokens = tokenizer.encode(token, add_special_tokens=False).tokens
                else:
                    sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_positions = [orig_to_tok_index[x] for x in start_positions]
            tok_end_positions = []
            for x in end_positions:
                if x < len(context) - 1:
                    tok_end_positions.append(orig_to_tok_index[x + 1] - 1)
                else:
                    tok_end_positions.append(len(all_doc_tokens) - 1)

            truncation = TruncationStrategy.ONLY_SECOND.value
            padding_strategy = "do_not_pad"
            encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                truncated_query,
                all_doc_tokens,
                truncation=truncation,
                padding=padding_strategy,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                return_token_type_ids=True,
            )
            tokens = encoded_dict['input_ids']
            type_ids = encoded_dict['token_type_ids']
            if args.simple:
                attn_mask = None
            else:
                attn_mask = encoded_dict['attention_mask']
            tokens_length = len(tokens)

            # find new start_positions/end_positions, considering
            # 1. we add query tokens at the beginning
            # 2. special tokens
            doc_offset = len(truncated_query) + sequence_added_tokens
            new_start_positions = [x + doc_offset for x in tok_start_positions if
                                   (x + doc_offset) < max_seq_length - 1]
            new_end_positions = [x + doc_offset if (x + doc_offset) < max_seq_length - 1 else max_seq_length - 2 for
                                 x in tok_end_positions]
            new_end_positions = new_end_positions[:len(new_start_positions)]

            feature = MRCFeatures(
                    tokens,
                    attn_mask,
                    type_ids,
                    new_start_positions,
                    new_end_positions,
                    doc_offset,
                    len_query,
                    len_context,
                )
            output_queue.put(("{}".format(i_t), feature))
        else:
            logging.info('Quit worker')
            break

def reduce_process(output_queue, all_features):
    '''
    write the processed data to the e2p_and_e2c from output_queue
    :param output_queue:
    :param e2p_and_e2c:
    :return:
    '''
    while True:
        i_t, features_one = output_queue.get()
        if features_one is None:
            logging.info('Quit Reducer')
            break
        else:
            all_features[i_t] = features_one

def cache_examples(args, tokenizer, evaluate=False):
    '''
    convert wikipedia data into the MRC training instances
    :param args:
    :return:
    '''
    process_count = max(1, args.processes)
    number_reducer = min(5, process_count // 4)

    if args.sample_data != "":
        sample_data = "_" + args.sample_data

    e2p_addr = os.path.join(args.data_path, "e2p" + ".bz2")
    if args.do_negative:
        e2c_train_addr = os.path.join(args.data_path, "e2c" + sample_data + "-negative" + ".train.bz2")
        e2c_test_addr = os.path.join(args.data_path, "e2c" + sample_data + "-negative" + ".test.bz2")
    else:
        e2c_train_addr = os.path.join(args.data_path, "e2c" + sample_data + ".train.bz2")
        e2c_test_addr = os.path.join(args.data_path, "e2c" + sample_data + ".test.bz2")
    e2p_ori = read_bz2(e2p_addr)
    if os.path.exists(e2c_train_addr) and os.path.exists(e2c_test_addr):
        logging.info("load from existing file")
        e2c_addr = e2c_test_addr if evaluate else e2c_train_addr
        e2c = read_bz2(e2c_addr)
    else:
        logging.info("create new file")
        e2c_addr = os.path.join(args.data_path, "e2c" + sample_data + ".bz2")
        e2c = read_bz2(e2c_addr)
        if args.do_negative:
            e2c_all = read_bz2(os.path.join(args.data_path, "e2c" + ".bz2"))
            e2c = add_negative(e2c_all=e2c_all, e2c=e2c)
            del e2c_all
        total_entities = list(e2c.keys())
        test_entities = random.sample(total_entities, k=args.PMR_test)
        test_entities = {x:0 for x in test_entities}
        e2c_train = {}
        e2c_test = {}
        for ix, x in enumerate(e2c):
            if x in test_entities:
                e2c_test[x] = e2c[x]
            else:
                e2c_train[x] = e2c[x]

        with bz2.BZ2File(e2c_train_addr, 'w') as writer:
            for item in tqdm(e2c_train.items(), 'save e2c train'):
                output_item = json.dumps(item, ensure_ascii=False) + "\n"
                writer.write(output_item.encode("utf-8"))
        with bz2.BZ2File(e2c_test_addr, 'w') as writer:
            for item in tqdm(e2c_test.items(), 'save e2c test'):
                output_item = json.dumps(item, ensure_ascii=False) + "\n"
                writer.write(output_item.encode("utf-8"))
        e2c = e2c_test if evaluate else e2c_train
    # prune e2p
    keep_keys = {}
    all_keys = list(e2c.keys())
    for key in all_keys:
        context = e2c[key]
        keep_keys[key] = 1
        for x in context:
            keep_keys[x[0]] = 1
    e2p = {}
    for key in keep_keys:
        e2p[key] = e2p_ori.pop(key)
    del e2p_ori

    total_pairs = 0
    for target in all_keys:
        total_pairs += len(e2c[target])
    if args.buffer != 0:
        buffer_num = math.ceil(total_pairs / args.buffer)
    else:
        buffer_num = 1
    logging.info("We have {} query-context pairs in total.".format(total_pairs))
    logging.info("Each buffer size is {}.".format(args.buffer))
    logging.info("The corpus will be saved into {} buffer files.".format(buffer_num))
    total_features = 0
    total_length = 0
    n_pairs = 0
    for i_buffer in range(buffer_num):
        buffer_keys = []
        if i_buffer == buffer_num - 1:
            buffer_keys = all_keys
        else:
            logging.info("i_buffer: {}".format(i_buffer))
            while n_pairs < (i_buffer + 1) * args.buffer:
                target = all_keys.pop(0)
                buffer_keys.append(target)
                n_pairs += len(e2c[target])

        logging.info("number of pairs : {}".format(n_pairs))
        if buffer_keys == []:
            continue
        input_data = [[x] + y for x in buffer_keys for y in e2c[x]]
        len_data = len(input_data)
        maxsize = 10000
        # output queue
        output_queue = Queue(maxsize=maxsize)

        manager = Manager()
        all_features = manager.dict()

        worker_count = process_count
        reduces = []
        # reduce job that sorts and prints output
        for i in range(number_reducer):
            reduce = Process(target=reduce_process,
                             args=(output_queue, all_features))
            reduce.start()
            reduces.append(reduce)
        # initialize jobs queue
        jobs_queue = Queue(maxsize=maxsize)

        # start worker processes
        logging.info("Using %d worker processes.", process_count)
        logging.info("We will process %d entities.", len(buffer_keys))
        time_start = time.time()
        workers = []
        for i in range(worker_count - number_reducer):

            worker = Process(target=create_mrc,
                                args=(args, e2p, len_data, tokenizer, jobs_queue, output_queue, time_start))
            worker.daemon = True  # only live while parent process lives
            worker.start()
            workers.append(worker)

        # Mapper process
        for i_t, items in enumerate(tqdm(input_data, desc="assign jobs")):
                job = (i_t, items)
                jobs_queue.put(job) # goes to any available extract_process

        # signal termination
        for _ in workers:
            jobs_queue.put((None, None))
        # wait for workers to terminate
        for w in workers:
            w.join()

        # signal end of work to reduce process
        for _ in reduces:
            output_queue.put((None, None))
        # wait for it to finish
        for r in reduces:
            r.join()

        all_features = dict(all_features)
        for x in all_features:
            total_length += all_features[x].len_query
            total_length += all_features[x].len_context


        work_duration = time.time() - time_start
        total_features += len(all_features)
        logging.info("Finished %d-process in %.1fs ", process_count, work_duration)

        cached_features_file = os.path.join(
            args.data_path,
            "cached_{}_{}_{}_{}_{}_{}".format(
                "PMR",
                "train" if not evaluate else "test",
                list(filter(None, args.model_type.split("/"))).pop(),
                str(args.max_query_length),
                str(args.max_seq_length),
                args.sample_data,
            ),
        )
        if args.do_negative:
            cached_features_file = cached_features_file + "-negative"
        cached_features_file = cached_features_file + "_" + str(i_buffer)
        logging.info("save cached features")
        torch.save(all_features, cached_features_file)
        logging.info("Done")
        del all_features
    logging.info('there are %d words in the corpus.', total_length)
    logging.info("Totally, we can get %d features as %s instances", total_features, 'training' if not evaluate else "test")
if __name__ == "__main__":
    default_process_count = max(1, cpu_count() - 1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=default_process_count,
                        help="Number of processes to use (default %(default)s)")
    parser.add_argument("--file", type=str, default="./en",
                        help="wikipedia dump file directory")
    parser.add_argument("--window", type=int, default=2,
                        help="window size to select former and later passages as context")
    parser.add_argument("--def_num", type=int, default=1,
                        help="we use the first def_num sentences as the definition for each entity")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.",)
    parser.add_argument("--max_query_length", type=int, default=128,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.",)
    parser.add_argument("--model_type", type=str, default="roberta",
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES))
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Where do you want to store the pre-trained models downloaded from huggingface.co")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--PMR_test", type=int, default=1000,
                        help="generate test set with size of PMR_test",)
    parser.add_argument("--buffer", type=int, default=8000000,
                        help="generate test set with size of PMR_test",)
    parser.add_argument("--sample_data", type=str, default='10-10', help="the sampled file file for generating data")
    parser.add_argument("--evaluate", action="store_true",
                        help="generate test or train set.")
    parser.add_argument("--do_negative", action="store_true",
                        help="if add negative examples.")
    parser.add_argument("--simple", action="store_true",
                        help="if true, do not save attn_mask and type_ids for saving memory.")
    args = parser.parse_args()
    if args.model_name_or_path == 'bert-base-uncased':
        assert args.do_lower_case
    args.data_path = os.path.join(args.file, "processed")
    args.max_context_length = args.max_seq_length - args.max_query_length
    args.model_type = args.model_type.lower()
    config = config_PMR[args.model_type].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_PMR[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,
        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )

    if args.evaluate:
        cache_examples(args, tokenizer=tokenizer, evaluate=True)
    else:
        cache_examples(args, tokenizer=tokenizer, evaluate=False)