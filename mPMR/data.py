#!/usr/bin/env python3
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers.utils import logging
logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet", 'xlmroberta'}


class PMRDataset(Dataset):
    def __init__(self, data_path, tokenizer: None, max_length: int = 512, max_query_length = 64, pad_to_maxlen=False, context_first=False, evaluate=False, lazy_load=True):
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
                             tqdm(keys, desc="prepare dataset at buffer {}".format(self.buffer_id)) if all(
                 features[x].start_positions[i] <= features[x].end_positions[i] for i, _ in
                 enumerate(features[x].start_positions))]

        else:
            self.all_data = []
            for buffer_id in range(len(self.all_buffer_files)):
                features = torch.load(self.all_buffer_files[buffer_id])
                keys = list(features.keys())
                for x in tqdm(keys, desc="prepare dataset at buffer {}".format(buffer_id)):
                    feature_one = features.pop(x)
                    if all(feature_one.start_positions[i] <= feature_one.end_positions[i] for i, _ in
                 enumerate(feature_one.start_positions)):
                        self.all_data.append(feature_one)
            self.all_buffer_files = [1] # set the length of self.all_buffer_files to be 1

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.pad_to_maxlen = pad_to_maxlen
        self.context_first = context_first

    def next_buffer(self):
        del self.all_data
        self.buffer_id = (self.buffer_id + 1) % len(self.all_buffer_files)
        features = torch.load(self.all_buffer_files[self.buffer_id])
        keys = list(features.keys())
        self.all_data = [features.pop(x) for x in
                         tqdm(keys, desc="prepare dataset at buffer {}".format(self.buffer_id)) if all(
             features[x].start_positions[i] <= features[x].end_positions[i] for i, _ in
             enumerate(features[x].start_positions))]
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
        sequence_added_tokens = (
            self.tokenizer.model_max_length - self.tokenizer.max_len_single_sentence + 1
            if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
            else self.tokenizer.model_max_length - self.tokenizer.max_len_single_sentence
        )
        if self.context_first:
            label_mask = [1] + [1] * (offset - sequence_added_tokens) + [0] * (seq_len - offset + sequence_added_tokens - 1) # allow cls in DLM loss
        else:
            label_mask = [1] + [0] * (offset - 1) + [1] * (seq_len - offset - 1) + [0]

        assert all(data.start_positions[i] <= data.end_positions[i] for i, _ in enumerate(data.start_positions))

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