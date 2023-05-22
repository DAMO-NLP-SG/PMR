
import os
import bz2
import json
from tqdm import tqdm
from multiprocessing import Queue, Process, cpu_count, Manager
import argparse
import logging
import time
from urllib.parse import unquote
import re
import random
from opencc import OpenCC
import spacy
from sacremoses import MosesTokenizer
from pythainlp import word_tokenize as thai_tokenizer
random.seed(42)
logging.getLogger().setLevel(logging.INFO)

href_pattern = r'<a.*?href=\"(.*?)\".*?>(.+?)</a>'
href_pattern_bad = r'<a.*?href=\"(.*?)\".*?></a>'
placeholder = "IAmAPlaceholder"
nonspace_lang = ['zh', 'ja', 'th']
spacy_lang = ['de', 'en', 'es', 'fr', 'it', 'ja', 'pl', 'pt', 'ru', 'zh', 'nl', 'fi', 'el', 'ko', 'sv']
lang2nlp = {
    "de": "de_core_news_md",  # 42m
    "en": "en_core_web_md",  # 40m
    "es": "es_core_news_md",  # 40m
    "fr": "fr_core_news_md",  # 43m
    "it": "it_core_news_md",  # 40m
    "ja": "ja_core_news_md",  # 39m
    "pl": "pl_core_news_md",  # 47m
    "pt": "pt_core_news_md",  # 40m
    "ru": "ru_core_news_md",  # 39m
    "zh": "zh_core_web_md",  # 74m
    "nl": "nl_core_news_md",  # 40m
    "fi": "fi_core_news_md",  # 65m
    "el": "el_core_news_md",  # 40m
    "ko": "ko_core_news_md",  # 65m
    "sv": "sv_core_news_md",  # 63m
}

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or c == "\n\n" or (len(c) == 1 and ord(c) == 0x202F):
        return True
    return False

def merge_e2c(src, tgt):
    for key in list(set(tgt)):
        if src.get(key):
            src[key] = src[key] + tgt.pop(key)
        else:
            src.update({key: tgt.pop(key)})
    return src

def filter_emoji(desstr, restr=''):
    # 过滤表情
    res = re.compile(u'[\U00010000-\U0010ffff\\uD800-\\uDBFF\\uDC00-\\uDFFF]')
    return res.sub(restr, desstr)

def check_disambiguation(input_text, entity, lang='en'):
    if lang == "ar" and (not "(توضيح)" in entity) and (not (input_text.endswith("إلى:") or input_text.endswith("إلى"))):
        return True
    elif lang == 'de' and (not "(Begriffsklärung)" in entity) and (not (input_text.endswith("steht für:") or input_text.endswith("steht für"))):
        return True
    elif lang == "en" and (not "(disambiguation)" in entity) and (not (input_text.endswith("refer to:") or input_text.endswith("refer to"))):
        return True
    elif lang == 'es' and (not "(desambiguación)" in entity) and (not (input_text.endswith("se puede referir a:") or input_text.endswith("se puede referir a"))):
        return True
    elif lang == 'fr' and (not "(homonymie)" in entity) and (not (input_text.endswith("peut désigner :") or input_text.endswith("peut désigner ") or input_text.endswith("peut désigner:"))):
        return True
    elif lang == 'it' and (not "(disambigua)" in entity):
        return True
    elif lang == 'ja' and (not "(曖昧さ回避)" in entity):
        return True
    elif lang == 'pl' and (not "(kategoria Strony ujednoznaczniające)" in entity):
        return True
    elif lang == 'pt' and (not "(desambiguação)" in entity) and (not (input_text.endswith("referir-se a:") or input_text.endswith("referir-se a"))):
        return True
    elif lang == 'ru' and (not "(значения)" in entity) and (not (input_text.endswith("может означать:") or input_text.endswith("может означать"))):
        return True
    elif lang == 'zh' and (not "(消歧义)" in entity) and (not (input_text.endswith("也指：") or input_text.endswith("可以指：") or input_text.endswith("可以指"))):
        return True
    elif lang == 'sv' and (not "(olika betydelser)" in entity) and (not (input_text.endswith("kan syfta på:") or input_text.endswith("kan syfta på"))):
        return True
    elif lang == 'tr' and (not "(anlam ayrımı)" in entity) and (not (input_text.endswith("şu anlamlara gelebilir:") or input_text.endswith("şu anlamlara gelebilir"))):
        return True
    elif lang == 'el' and (not "(αποσαφήνιση)" in entity) and (not (input_text.endswith("μπορεί να αναφέρονται:") or input_text.endswith("μπορεί να αναφέρονται"))):
        return True
    elif lang == 'ko' and (not "(분류 동음이의어 문서)" in entity) and (not (input_text.endswith("다음과 같은 동음이의어가 있다.") or input_text.endswith("다음과 같은 동음이의어가 있다"))):
        return True
    elif lang == 'nl' and (not "(doorverwijspagina)" in entity) and (not (input_text.endswith("verwijzen naar:") or input_text.endswith("verwijzen naar"))):
        return True
    elif lang == 'id' and (not "(disambiguasi)" in entity) and (not (input_text.endswith("dapat mengacu pada beberapa hal berikut:") or input_text.endswith("dapat mengacu pada beberapa hal berikut"))):
        return True
    elif lang == 'fi' and (not "(täsmennyssivu)" in entity) and (not (input_text.endswith("voi tarkoittaa seuraavia asioita:") or input_text.endswith("voi tarkoittaa seuraavia asioita"))):
        return True
    elif lang == 'vi' and (not "(định hướng)" in entity) and (not (input_text.endswith("có thể đề cập đến:") or input_text.endswith("có thể đề cập đến"))):
        return True
    elif lang == 'bn' and (not "(দ্ব্যর্থতা নিরসন)" in entity) and (not (input_text.endswith("বোঝানো হতে পারে -") or input_text.endswith("বোঝানো হতে পারে-") or input_text.endswith("বোঝানো হতে পারে;") or input_text.endswith("বোঝানো হতে পারে। যথা:") or input_text.endswith("বোঝানো হতে পারে:"))):
        return True
    elif lang == 'hi' and (not "(बहुविकल्पी)" in entity) and (not (input_text.endswith("कई अर्थ हो सकते हैं:") or input_text.endswith("कई अर्थ हो सकते हैं"))):
        return True
    elif lang == 'th' and (not "(แก้ความกำกวม)" in entity) and (not (input_text.endswith("อาจหมายถึง"))):
        return True
    elif lang == 'te' and (not "(అయోమయ నివృత్తి)" in entity):
        return True
    elif lang == 'sw':
        return True
    else:
        return False

def read_bz2(args, addr, unique_entity):
    if args.lang in spacy_lang:
        nlp = spacy.load(lang2nlp[args.lang],
                         exclude=["tok2vec", "tagger", "parser", 'senter', "attribute_ruler", "lemmatizer", 'ner'])
    elif args.lang == 'th':
        tokenizer = thai_tokenizer
    else:
        tokenizer = MosesTokenizer(lang=args.lang)
    with bz2.open(addr) as reader:
        e2p_one = {}
        e2c_one = {}
        for line in reader:
            js_line = json.loads(line)
            entity = js_line['title']
            passage = filter_emoji(js_line['text'].strip())
            if (args.lang == 'id' and (entity == 'Teks Bizantin' or entity =='Tōyō Kanji')) or (args.lang == 'nl' and entity == 'Lijst van Nederlandse voetballers'): #special cases
                continue
            if args.lang == 'zh':
                t2s = OpenCC('t2s')
                passage = t2s.convert(passage)
                entity = t2s.convert(entity)
            passage = re.sub(href_pattern_bad, "", passage)  # remove reference citation e.g. 还可以减少癫痫病患者的发病的机率[1]
            all_anchors = re.findall(href_pattern, passage)
            passage_mask = re.sub(href_pattern, " " + placeholder + " ", passage)
            if len(passage_mask.split("\n\n", 1)) < 2 or entity in unique_entity:
                continue
            passage_mask = passage_mask.split("\n\n", 1)[1] # remove the repeated title
            passage_mask = " ".join(passage_mask.split())
            # 有些page是重定向页面 may refer to:\n
            if check_disambiguation(passage_mask, entity, args.lang):
                if args.lang not in spacy_lang and args.lang not in nonspace_lang:
                    passage_mask_token = tokenizer.tokenize(passage_mask, escape=False)
                elif args.lang in spacy_lang and args.lang not in nonspace_lang:
                    passage_mask_token = [x.text for x in nlp(passage_mask)]
                elif args.lang == 'th':
                    passage_mask_token = tokenizer(passage_mask, keep_whitespace=False)
                else:
                    passage_segment = passage_mask.split(placeholder)
                    passage_mask_token = []
                    for ix, x in enumerate(passage_segment):
                        x = x.strip()
                        if x == "" and ix < len(passage_segment) - 1:
                            passage_mask_token.append(placeholder)
                        else:
                            if args.lang == 'ja' and len(x) > 5000: # ja tokenizer cannot process extremely long document
                                segment_one = []
                                x_split = x.split("。")
                                for y in x_split:
                                    if y.strip() != "":
                                        segment_one += [z.text for z in nlp(y)] + ["。"]
                            else:
                                segment_one = [y.text for y in nlp(x)]
                            passage_mask_token.extend(segment_one)
                            if ix < len(passage_segment) - 1:
                                passage_mask_token.append(placeholder)


                assert passage_mask_token.count(placeholder) == len(all_anchors)
                passage_mask_token = [x for x in passage_mask_token if not _is_whitespace(x)]
                # remove short article
                if len(passage_mask_token) <= 20:
                    continue

                new_passage = []
                num_ph = 0
                for i_t, tok in enumerate(passage_mask_token):
                    if tok == placeholder:
                        anchor = all_anchors[num_ph]
                        anchor_entity = unquote(anchor[0])
                        if args.lang == 'zh':
                            anchor_entity = t2s.convert(anchor_entity)
                        if args.lang in spacy_lang:
                            anchor_tok = [x.text for x in nlp(anchor[1])]
                        elif args.lang == 'th':
                            anchor_tok = tokenizer(anchor[1], keep_whitespace=False)
                        else:
                            anchor_tok = tokenizer.tokenize(anchor[1], escape=False)
                        anchor_start = len(new_passage)
                        anchor_end = anchor_start + len(anchor_tok) - 1
                        new_passage.extend(anchor_tok)
                        if not anchor_entity.startswith("#") and anchor_entity != "":
                            # \# indicates the self anchor to the section name, "" is noncharacter
                            if anchor_start <= anchor_end:
                                if anchor_entity not in e2c_one:
                                    e2c_one[anchor_entity] = []
                                e2c_one[anchor_entity].append((entity, anchor_start, anchor_end))
                        num_ph += 1
                    else:
                        new_passage.append(tok)

                assert num_ph == len(all_anchors)
                if entity not in e2p_one:
                    e2p_one[entity] = new_passage
                else:
                    logging.info("Two entities have the same title @ {}".format(addr),)
    return e2p_one, e2c_one

def load_one_dir(args, parent_addr, jobs_queue, output_queue, unique_entity):
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
            for one_file in tqdm(all_files, desc="reading file at {}".format(job)):
                file_path = os.path.join(dir_path, one_file)
                e2p_one, e2c_one = read_bz2(args, file_path, unique_entity)
                e2p.update(e2p_one)
                e2c = merge_e2c(e2c, e2c_one)
            output_queue.put((job, e2p, e2c))
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
        job, e2p, e2c = output_queue.get()
        if job is None:
            logging.info('Quit Reducer')
            break
        else:
            logging.info('add result at {}'.format(job))
            e2p_and_e2c[job] = (e2p, e2c)

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
    unique_entity = manager.dict()
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
                            args=(args, parent_addr, jobs_queue, output_queue, unique_entity))
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
    args.lang = args.file.split('/')[-1]
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