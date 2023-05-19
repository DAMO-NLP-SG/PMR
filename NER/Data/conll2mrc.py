#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  {
    "context": "Xinhua News Agency , Shanghai , August 31st , by reporter Jierong Zhou",
    "end_position": [
      2
    ],
    "entity_label": "ORG",
    "impossible": false,
    "qas_id": "0.2",
    "query": "organization entities are limited to companies, corporations, agencies, institutions and other groups of people.",
    "span_position": [
      "0;2"
    ],
    "start_position": [
      0
    ]
  }
"""

import os
import json

def normalize_word(word, language='english'):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def get_original_token(token):
    escape_to_original = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    if token in escape_to_original:
        token = escape_to_original[token]
    return token

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
            token_data_line = get_original_token(token_data_line)
            token_data_line = normalize_word(token_data_line)
            cached_token.append(token_data_line)
            cached_label.append(label_data_line)
    new_data = []
    for x in dataset_item_lst:
        if x[0] != []:
            new_data.append(x)
    return new_data

def conll2mrc(conll_f):
    count = 0
    label_details = {'ORG': "organization entities are limited to named corporate, governmental, or other organizational entities.",
                     'PER': "person entities are named persons or family.",
                     'LOC': "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
                     'MISC': "examples of miscellaneous entities include events, nationalities, products and works of art.",
                     }
    mrc_f = []
    all_label_dict = []
    for ic, one in enumerate(conll_f):
        context = one[0]
        label_list = one[1]
        label_dict = {x:[] for x in label_details}
        last_label = None
        for il, label_one in enumerate(label_list):
            if last_label is not None and last_label.startswith("I") and not label_one.startswith("I"):
                assert label_id_start == label_id_end
                assert label_start < label_end
                label_dict[label_id_start].append([label_start, label_end])
            elif last_label is not None and last_label.startswith("B") and not label_one.startswith("I"):
                label_dict[label_id_start].append([label_start, label_start])
            if label_one.startswith("B"):
                count += 1
                label_id_start = label_one.split(".")[0].split("-")[1]
                label_start = il
                if il == len(label_list) - 1:
                    label_dict[label_id_start].append([il, il])
            elif label_one.startswith("I"):
                label_id_end = label_one.split(".")[0].split("-")[1]
                label_end = il
                if il == len(label_list) - 1:
                    assert label_id_start == label_id_end
                    assert label_start < label_end
                    label_dict[label_id_start].append([label_start, label_end])


            last_label = label_one
        all_label_dict.append(label_dict)
        for il, label_id in enumerate(label_details):
            entity_label = label_id
            query = label_details[label_id]
            qas_id = '{}.{}'.format(ic, il)
            label_item = label_dict[label_id]
            start_position = [x[0] for x in label_item]
            end_position = [x[1] for x in label_item]
            span_position = ['{};{}'.format(x[0], x[1]) for x in label_item]
            if span_position != []:
                impossible = False
            else:
                impossible = True
            mrc_one = {
                "context": " ".join(context),
                "end_position": end_position,
                "entity_label": entity_label,
                "impossible": impossible,
                "qas_id": qas_id,
                "query": query,
                "span_position": span_position,
                "start_position": start_position,
            }
            mrc_f.append(mrc_one)
    return mrc_f


if __name__ == "__main__":
    for fewshot in ['-few5-1', '-few5-2', '-few5-3', '-few10-1', '-few10-2', '-few10-3', '-few20-1', '-few20-2', '-few20-3', '-few50-1', '-few50-2', '-few50-3']:
        data_file_path = os.path.join("./conll03" + fewshot, "tag-ner.train")
        conll_f = read_conll(data_file_path, delimiter=" ")
        mrc_f = conll2mrc(conll_f)
        save_file = os.path.join("./conll03" + fewshot, "mrc-ner.train")
        with open(save_file, 'w') as writer:
            json.dump(mrc_f,writer, ensure_ascii=False, sort_keys=True, indent=2)
# maxlen = 0
# for one in mrc_f:
#         maxlen += len(one['start_position'])
# label_dict = {}
# for one in dataset_item_lst:
#     for label in one[1]:
#         if label.startswith("B"):
#             label_name = label.replace("B-", "")
#             if label_name not in label_dict:
#                 label_dict[label_name] = 0
#             label_dict[label_name] += 1