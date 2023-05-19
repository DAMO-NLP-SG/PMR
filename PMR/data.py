#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers.utils import logging
logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


class PMRDataset(Dataset):

    def __init__(self, data_path, tokenizer: None, max_length: int = 512, max_query_length = 64, pad_to_maxlen=False, evaluate=False, lazy_load=True):
        drive, tail = os.path.split(data_path)
        self.all_buffer_files = sorted([os.path.join(drive, x) for x in os.listdir(drive) if x.startswith(tail)], key= lambda x: int(x.split("_")[-1]))
        self.buffer_id = 0
        self.lazy_load = lazy_load
        if evaluate:
            if len(self.all_buffer_files) != 1:
                logger.error("please save evaluate file in one single buffer file")
                assert len(self.all_buffer_files) == 1
        if self.lazy_load:
            features = torch.load(self.all_buffer_files[self.buffer_id])
            keys = list(features.keys())
            self.all_data = [features.pop(x) for x in
                             tqdm(keys, desc="prepare dataset at buffer {}".format(self.buffer_id))]
        else:
            self.all_data = []
            for buffer_id in range(len(self.all_buffer_files)):
                features = torch.load(self.all_buffer_files[buffer_id])
                keys = list(features.keys())
                for x in tqdm(keys, desc="prepare dataset at buffer {}".format(buffer_id)):
                    feature_one = features.pop(x)
                    self.all_data.append(feature_one)
            self.all_buffer_files = [1] # set the length of self.all_buffer_files to be 1

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.pad_to_maxlen = pad_to_maxlen

    def next_buffer(self):
        del self.all_data
        self.buffer_id = (self.buffer_id + 1) % len(self.all_buffer_files)
        features = torch.load(self.all_buffer_files[self.buffer_id])
        keys = list(features.keys())
        self.all_data = [features.pop(x) for x in tqdm(keys, desc="prepare dataset at buffer {}".format(self.buffer_id))]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            attention_mask: attention mask, 1 for token, 0 for padding, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
        """
        data = self.all_data[item]
        seq_len = len(data.input_ids)

        pair_input_ids = data.input_ids
        if data.attention_mask is None:
            pair_attention_mask = [1] * len(pair_input_ids)
        else:
            pair_attention_mask = data.attention_mask
        if data.token_type_ids is None:
            pair_token_types_ids = [0] * len(pair_input_ids)
        else:
            pair_token_types_ids = data.token_type_ids
        offset = data.doc_offset
        tokenizer_type = type(self.tokenizer).__name__.replace("Tokenizer", "").lower()

        label_mask = [1] + [0] * (offset - 1) + [1] * (seq_len - offset - 1) + [0]

        assert all(label_mask[p] != 0 for p in data.start_positions)
        assert all(label_mask[p] != 0 for p in data.end_positions)
        assert len(label_mask) == seq_len

        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        if data.start_positions != [] and data.end_positions != []:
            match_labels[0, 0] = 1
        for start, end in zip(data.start_positions, data.end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(pair_input_ids),
            torch.LongTensor(pair_attention_mask),
            torch.LongTensor(pair_token_types_ids),
            torch.LongTensor(label_mask),
            match_labels,
        ]

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)