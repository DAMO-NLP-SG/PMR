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
        if data_line.startswith("-DOCSTART-"):
            continue
        if idx != 0 and len(data_line) == 0:
            dataset_item_lst.append([cached_token, cached_label])
            cached_token, cached_label = [], []
        else:
            token_label = data_line.split(delimiter)
            token_data_line, label_data_line = token_label[0], token_label[1]
            if label_data_line.startswith("M"):
                label_data_line = "I" + label_data_line[1:]
            elif label_data_line.startswith("E"):
                label_data_line = "I" + label_data_line[1:]
            elif label_data_line.startswith("S"):
                label_data_line = "B" + label_data_line[1:]
            token_data_line = get_original_token(token_data_line)
            token_data_line = normalize_word(token_data_line)
            cached_token.append(token_data_line)
            cached_label.append(label_data_line)
    new_data = []
    for x in dataset_item_lst:
        if x[0] != []:
            new_data.append(x)
    return new_data


def conll2mrc(conll_f, lang):
    count = 0
    if lang == "en":
        label_details = {'POS': "for aspect terms of positive sentiment.",
                         'NEG': "for aspect terms of negative sentiment.",
                         'NEU': "for aspect terms of neutral sentiment.",
                         }
    elif lang == "fr":
        label_details = {'POS': "pour les termes d'aspect du sentiment positif.",
                         'NEG': "pour les termes d'aspect du sentiment négatif.",
                         'NEU': "pour les termes d'aspect du sentiment neutre.",
                         }
    elif lang == "es":
        label_details = {'POS': "para términos de aspecto de sentimiento positivo.",
                         'NEG': "para términos de aspecto de sentimiento negativo.",
                         'NEU': "para términos de aspecto de sentimiento neutral.",
                         }
    elif lang == "nl":
        label_details = {'POS': "voor aspect termen van positief sentiment.",
                         'NEG': "voor aspect termen van negatief sentiment.",
                         'NEU': "voor aspect termen van neutraal sentiment.",
                         }
    elif lang == "ru":
        label_details = {'POS': "для аспектов положительного настроения.",
                         'NEG': "для аспектов отрицательного настроения.",
                         'NEU': "для аспектов нейтрального настроения.",
                         }
    elif lang == "tr":
        label_details = {'POS': "Pozitif duyguların görünüş açısından.",
                         'NEG': "olumsuz duyguların en boy terimleri için.",
                         'NEU': "nötr duygunun görünüş terimleri için.",
                         }
    else:
        raise NotImplementedError


    mrc_f = []
    all_label_dict = []
    for ic, one in enumerate(conll_f):
        context = one[0]
        label_list = one[1]
        label_dict = {x:[] for x in label_details}
        last_label = None
        for il, label_one in enumerate(label_list):

            if last_label is not None and last_label.startswith("T") and not label_one.startswith("T"):
                assert label_start <= il - 1
                label_dict[label_id_start].append([label_start, il - 1])
            if last_label is not None and last_label.startswith("T") and label_one.startswith("T"):
                assert label_start <= il
                if il == len(label_list) - 1:
                    label_dict[label_id_start].append([label_start, il])
            elif (last_label is None or not last_label.startswith("T")) and label_one.startswith("T"):
                count += 1
                label_id_start = label_one.split("-")[1]
                label_start = il
                if il == len(label_list) - 1:
                    label_dict[label_id_start].append([il, il])
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
    for dataset in ['train', 'dev', 'test']:
        for lang in ['en', 'fr', 'es', 'nl', 'ru', 'tr']:
            data_file_path = os.path.join("./rest", 'gold-' + lang + "-" + dataset + ".txt")
            conll_f = read_conll(data_file_path, delimiter="\t")

            mrc_f = conll2mrc(conll_f, lang=lang)
            save_file = os.path.join("./rest", lang + "-mrc-absa" + "." + dataset)
            with open(save_file, 'w') as writer:
                json.dump(mrc_f, writer, ensure_ascii=False, sort_keys=True, indent=2)
