
import os
import bz2
import json
from tqdm import tqdm
from multiprocessing import Queue, Process, cpu_count, Manager
import argparse
import logging
import time
from nltk import word_tokenize, sent_tokenize
from urllib.parse import unquote
import re
import random
random.seed(42)
logging.getLogger().setLevel(logging.INFO)

href_pattern = r'<a.*?href=\"(.*?)\".*?>(.*?)</a>'
placeholder = "IAmAPlaceholder"

def merge_e2c(src, tgt):
    for key in list(set(tgt)):
        if src.get(key):
            src[key] = src[key] + tgt.pop(key)
        else:
            src.update({key: tgt.pop(key)})
    return src

def read_bz2(addr):
    with bz2.open(addr) as reader:
        e2p_one = {}
        e2c_one = {}
        for line in reader:
            js_line = json.loads(line)
            # 有些page是重定向页面 may refer to:\n
            striped_text = js_line['text'].strip()
            if not (striped_text.endswith("may refer to:") or striped_text.strip().endswith("may also refer to:") or len(striped_text) <= 500):
                passage = sent_tokenize(striped_text)
                entity = js_line['title']
                new_passage = []
                for i_s, sen_one in enumerate(passage):
                    all_anchors = re.findall(href_pattern, sen_one)
                    sen_one_nohref = re.sub(href_pattern, " " + placeholder + " ", sen_one)
                    sen_tok = word_tokenize(sen_one_nohref)
                    num_ph = 0
                    i_t = 0
                    while i_t < len(sen_tok):
                        tok = sen_tok[i_t]
                        if tok == placeholder:
                            anchor = all_anchors[num_ph]
                            anchor_entity = unquote(anchor[0])
                            anchor_tok = word_tokenize(anchor[1])
                            anchor_start = i_t
                            anchor_end = i_t + len(anchor_tok) - 1
                            sen_tok = sen_tok[:i_t] + anchor_tok + sen_tok[i_t + 1:]
                            if not anchor_entity.startswith("#") and anchor_entity != "":
                                # \# indicates the self anchor to the section name, "" is noncharacter
                                if anchor_entity not in e2c_one:
                                    e2c_one[anchor_entity] = []
                                if anchor_start <= anchor_end:
                                    e2c_one[anchor_entity].append((entity, i_s, anchor_start, anchor_end))
                            i_t += len(anchor_tok) - 1
                            num_ph += 1
                        i_t += 1
                    assert num_ph == len(all_anchors)
                    new_passage.append(sen_tok)

                if entity not in e2p_one:
                    e2p_one[entity] = new_passage
                else:
                    logging.info("Two entities have the same title",)
    return e2p_one, e2c_one

def load_one_dir(parent_addr, jobs_queue, output_queue):
    '''
    load Wiki data in each directory
    :param pro_id:
    :param parent_addr: directory name
    :param jobs_queue:
    :param output_queue:
    :return:
    '''
    while True:
        dir_id, job = jobs_queue.get()
        if job:
            logging.info('current job is %s', job)
            e2p = {}
            e2c = {}
            dir_path = os.path.join(parent_addr, job)
            all_files = os.listdir(dir_path)
            all_files = sorted([x for x in all_files if not x.startswith(".")])
            for one_file in all_files:
                file_path = os.path.join(dir_path, one_file)
                e2p_one, e2c_one = read_bz2(file_path)
                e2p.update(e2p_one)
                e2c = merge_e2c(e2c, e2c_one)
            output_queue.put((dir_id, e2p, e2c))
        else:
            logging.info('Quit worker')
            break

def reduce_process(output_queue, e2p_and_e2c):
    '''
    write the processed data to the e2p_and_e2c from output_queue
    :param output_queue:
    :param e2p_and_e2c:
    :return:
    '''
    while True:
        dir_id, e2p, e2c = output_queue.get()
        if dir_id is None:
            logging.info('Quit Reducer')
            break
        else:
            e2p_and_e2c[dir_id] = (e2p, e2c)

def read_wiki_multiprocess(args):
    parent_addr = args.file
    process_count = max(1, args.processes)

    all_dir = os.listdir(parent_addr)
    all_dir = sorted([x for x in all_dir if not x.startswith(".") and not x.endswith(".bz2") and not x == "processed"])

    maxsize = 10000
    # output queue
    output_queue = Queue(maxsize=maxsize)

    manager = Manager()
    e2p_and_e2c = manager.dict()
    worker_count = process_count
    # reduce job that sorts and prints output
    reduce = Process(target=reduce_process,
                     args=(output_queue, e2p_and_e2c))
    reduce.start()
    # initialize jobs queue
    jobs_queue = Queue(maxsize=maxsize)

    # start worker processes
    logging.info("Using %d worker processes.", process_count)
    time_start = time.time()
    workers = []
    for i in range(worker_count):
        worker = Process(target=load_one_dir,
                            args=(parent_addr, jobs_queue, output_queue))
        worker.daemon = True  # only live while parent process lives
        worker.start()
        workers.append(worker)

    # Mapper process
    page_num = 0
    for i_d, one_dir in enumerate(tqdm(all_dir, desc="assign jobs")):
            job = one_dir
            jobs_queue.put((i_d, job)) # goes to any available extract_process

    # signal termination
    for _ in workers:
        jobs_queue.put((None, None))
    # wait for workers to terminate
    for w in workers:
        w.join()

    # signal end of work to reduce process
    output_queue.put((None, None, None))
    # wait for it to finish
    reduce.join()
    new_e2p_and_e2c = {}

    new_e2p_and_e2c['e2p'] = {}
    new_e2p_and_e2c['e2c'] = {}
    dir_list = list(e2p_and_e2c.keys())
    for dir_id in tqdm(dir_list, 'iteration on merge file'):
        e2p_one, e2c_one = e2p_and_e2c.pop(dir_id)
        new_e2p_and_e2c['e2p'] = merge_e2c(new_e2p_and_e2c['e2p'], e2p_one)
        new_e2p_and_e2c['e2c'] = merge_e2c(new_e2p_and_e2c['e2c'], e2c_one)
    work_duration = time.time() - time_start
    logging.info("Finished %d-process in %.1fs ", process_count, work_duration)

    return new_e2p_and_e2c

def load_wiki(args):
    e2p_and_e2c = read_wiki_multiprocess(args)
    entity2passage = e2p_and_e2c['e2p']
    entity2context = e2p_and_e2c['e2c']
    logging.info("get e2p and e2c from %d process", len(e2p_and_e2c))
    return entity2passage, entity2context

if __name__ == "__main__":
    default_process_count = max(1, cpu_count() - 1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=default_process_count,
                        help="Number of processes to use (default %(default)s)")
    parser.add_argument("--file", type=str, default="./en",
                        help="wikipedia dump file directory")
    args = parser.parse_args()

    entity2passage, entity2context = load_wiki(args)

    common_entity = list(set(entity2context.keys()).intersection(set(entity2passage.keys())))
    logging.info("Ideally, we can get %d Wiki passages from Wiki, but only %d are found in e2c", len(entity2passage), len(common_entity))

    difference = list(set(entity2context.keys()).difference(set(entity2passage.keys())))
    if difference:
        logging.warning("there are %d entity anchors not existing in e2p, some examples are %s. We need to remove those.", len(difference), difference[:100])
        [entity2context.pop(k) for k in difference]
    num_instances = sum([len(entity2context[x]) for x in entity2context])
    logging.info("At most, we can get %d MRC instances from Wiki", num_instances)

    logging.info("save e2c and e2p.")
    save_dir = os.path.join(args.file, "processed")
    if not os.path.exists(save_dir):
       os.makedirs(save_dir)
    with bz2.BZ2File(os.path.join(save_dir, 'e2p.bz2'), 'w') as writer:
        for item in tqdm(entity2passage.items(), desc='save e2p'):
            output_item = json.dumps(item, ensure_ascii=False) + "\n"
            writer.write(output_item.encode("utf-8"))
    with bz2.BZ2File(os.path.join(save_dir, 'e2c.bz2'), 'w') as writer:
        for item in tqdm(entity2context.items(), 'save e2c'):
            output_item = json.dumps(item, ensure_ascii=False) + "\n"
            writer.write(output_item.encode("utf-8"))