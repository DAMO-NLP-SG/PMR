import json
import argparse
import os
import shutil
import random
import re
import torch
import data

class QAFeatures:
    '''
    QA features
    '''
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions,
        end_positions,
        doc_offset,
        qid,
        token_to_orig_map=None,
        context=None,
        context_str=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.doc_offset = doc_offset
        self.qid = qid
        self.token_to_orig_map = token_to_orig_map
        self.context = context
        self.context_str = context_str

def read_conll(input_file, delimiter=" "):
    """load ner dataset from CoNLL-format files."""
    dataset_item_lst = []
    with open(input_file, "r", encoding="utf-8") as r_f:
        datalines = r_f.readlines()

    cached_token, cached_label = [], []
    for idx, data_line in enumerate(datalines):
        data_line = data_line.strip()
        if idx != 0 and len(data_line) == 0:
            dataset_item_lst.append([cached_token, cached_label])
            cached_token, cached_label = [], []
        else:
            token_label = data_line.split(delimiter)
            token_data_line, label_data_line = token_label[0], token_label[1]
            cached_token.append(token_data_line)
            cached_label.append(label_data_line)
    return dataset_item_lst

def sample_train_NER(args):
    random.seed(args.seed)
    args.indir = os.path.join("./NER", "Data", args.dataset)
    args.outdir = os.path.join("./NER", "Data", args.dataset + "-few" + str(args.fewshot))

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        # directly copy test set
        shutil.copy2(os.path.join(args.indir, "mrc-ner" + ".dev"),
                     os.path.join(args.outdir, "mrc-ner" + ".dev"))
        shutil.copy2(os.path.join(args.indir, "mrc-ner" + ".test"),
                     os.path.join(args.outdir, "mrc-ner" + ".test"))
    train_f = os.path.join(args.indir,"mrc-ner" + ".train")
    all_data = json.load(open(train_f, encoding="utf-8"))

    first = True
    for i_d, data_one in enumerate(all_data):
        qas_id = data_one['qas_id']
        example_id, label_id = qas_id.split(".")
        if first:
            first_id = example_id
            first = False
        if example_id != first_id:
            break
    label_count = i_d
    example_count = len(all_data) // label_count
    capacity = {i:0 for i in range(label_count)}
    sample_index = list(range(example_count))
    random.shuffle(sample_index)
    if args.fewshot > 1:
        fewshot_number = int(args.fewshot)
        per_label = True
    else:
        fewshot_number = int(args.fewshot * example_count)
        per_label = False
    fewshot_data = []
    if per_label:
        for i_c,i in enumerate(sample_index):
            examples = all_data[ i * label_count: (i+1) * label_count]
            isFull = False
            for j in capacity:
                if capacity[j] + len(examples[j]['start_position']) > fewshot_number:
                    isFull = True
                    break
            if not isFull:
                for j in capacity:
                    capacity[j] = capacity[j] + len(examples[j]['start_position'])
                fewshot_data.extend(examples)
            end = True
            for j in capacity:
                if capacity[j] < fewshot_number:
                    end = False
                    break
            if end:
                break
    else:
        for i_c,i in enumerate(sample_index[:fewshot_number]):
            examples = all_data[ i * label_count: (i+1) * label_count]
            for j in capacity:
                capacity[j] = capacity[j] + len(examples[j]['start_position'])
            fewshot_data.extend(examples)
    print('total size of fewshot examples are {}'.format(len(fewshot_data) // label_count))
    print("capacity", capacity)
    with open(os.path.join(args.outdir, "mrc-ner"  + ".train"), 'w') as writer:
        json.dump(fewshot_data, writer, ensure_ascii=False, sort_keys=True, indent=2)
    return fewshot_data

def sample_train_QA(args):
    random.seed(args.seed)
    cached_features_file = "../Data/cached_" + args.dataset + "_train_roberta_64_384"
    features = torch.load(cached_features_file)
    sample_index = list(range(len(features)))
    random.shuffle(sample_index)
    if args.fewshot > 1:
        fewshot_number = int(args.fewshot)
    else:
        fewshot_number = int(args.fewshot * len(sample_index))
    qids = {}
    new_features = []
    for i in sample_index:
        feature_one = features[i]
        qid = feature_one.qid
        start_positions = feature_one.start_positions
        if qid not in qids and len(start_positions) != 0:
            qids[qid] = 1
            new_features.append(feature_one)
        if len(new_features) >= fewshot_number:
            break
    print('total size of fewshot examples are {}'.format(len(new_features)))
    cached_features_file_out = "../Data/cached_" + args.dataset + "_{}-{}".format(fewshot_number, args.seed) + "_train_roberta_64_384"
    torch.save(new_features, cached_features_file_out)


def sample_train_MCQA(args):
    random.seed(args.seed)

    args.indir = os.path.join("../Data", args.dataset)
    with open(os.path.join(args.indir, "train.json"), 'r') as reader:
        dataset = json.load(reader)
    sample_index = list(range(len(dataset)))
    random.shuffle(sample_index)
    if args.fewshot > 1:
        fewshot_number = int(args.fewshot)
    else:
        fewshot_number = int(args.fewshot * len(sample_index))
    args.outdir = os.path.join("../Data", args.dataset + "_{}-{}".format(fewshot_number, args.seed))
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        # directly copy test set
        shutil.copy2(os.path.join(args.indir, "dev.json"),
                     os.path.join(args.outdir, "dev.json"))
        shutil.copy2(os.path.join(args.indir, "test.json"),
                     os.path.join(args.outdir, "test.json"))

    new_dataset = []
    for i in sample_index:
        dataset_one = dataset[i]
        passage = dataset_one[0]
        questions = dataset_one[1]
        dataset_id = dataset_one[2]
        new_questions = random.sample(questions,k=1)
        new_dataset_one = [passage, new_questions, dataset_id]
        new_dataset.append(new_dataset_one)
        if len(new_dataset) >= fewshot_number:
            break
    print('total size of fewshot examples are {}'.format(len(new_dataset)))
    with open(os.path.join(args.outdir, "train.json"), 'w') as writer:
        json.dump(new_dataset, writer, ensure_ascii=False, sort_keys=True, indent=2)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
        default="QA",
        type=str,
        required=False,
        help="do few-shot study on which task",
    )
    parser.add_argument(
        "--tag",
        default=0,
        type=int,
        required=False,
        help="whether to convert tagging file",
    )
    parser.add_argument(
        "--fewshot",
        default=5,
        type=float,
        required=False,
        help="how many fewshot examples select for each category",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        required=False,
        help="random seed",
    )
    args = parser.parse_args()
    if args.task == "NER":
        args.dataset = 'conll03'
        sample_train_NER(args)
    elif args.task == "QA":
        args.dataset = 'SQuAD'
        sample_train_QA(args)
    elif args.task == "MCQA":
        args.dataset = 'DREAM'
        sample_train_MCQA(args)
    elif args.task == "SemEval":
        args.dataset = 'semeval21'
        task = 'semeval'






