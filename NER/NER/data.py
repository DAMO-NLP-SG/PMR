#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}

class MRCNERDataset(Dataset):
    def __init__(self, data, tokenizer: None, max_length: int = 512, max_query_length = 64, var=0):
        self.all_data = data
        self.var = var
        self.prompt()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.max_context_length = max_length - max_query_length

    def prompt(self):
        num_examples = len(self.all_data)
        num_impossible = len([1 for x in self.all_data if x["impossible"]])
        self.neg_ratio = (num_examples - num_impossible) / num_impossible
        new_datas = []
        for data in self.all_data:
            label = data['entity_label']
            details = data['query']
            context = data['context']
            start_positions = data["start_position"]
            end_positions = data["end_position"]
            words = context.split()
            assert len(words) == len(context.split(" "))
            if self.var == 0:
                query = '"{}". {}'.format(label, details) # ori
            elif self.var == 1:
                query = 'What are the "{}" entity, where {}'.format(label, details) # variant 1
            elif self.var == 2:
                query = 'Identify the spans (if any) related to "{}" entity. Details: {}'.format(label, details)  # variant 2
            elif self.var == 3:
                query = '"{}". {}'.format(label, details).replace(
                    "organization entities are limited to named corporate, governmental, or other organizational entities.",
                "Organization entity is a type of named entity that refers to a company, corporation, institution, or any other type of organization.").replace(
                    "person entities are named persons or family.",
                "Person entity is a type of named entity that refers to an individual's name or personal identifier, such as a celebrity, politician, or ordinary person.").replace(
                    "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
                "Location entity is a type of named entity that refers to a place or geographic location.").replace(
                    "examples of miscellaneous entities include events, nationalities, products and works of art.",
                'Miscellaneous entity can refer to things such as products, events, dates, times, currencies, quantities, and more.')
            elif self.var == 4:
                query = '"{}". {}'.format(label, details).replace(
                    "organization entities are limited to named corporate, governmental, or other organizational entities.",
                    "Organization entity typically includes information such as the organization's name, type, location, and industry.").replace(
                    "person entities are named persons or family.",
                    "Person entity typically includes information such as the first name, last name, and other personal identifiers such as title, age, gender, and occupation.").replace(
                    "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
                    "Location entity may include information such as the place's name, type, and geographic coordinates.").replace(
                    "examples of miscellaneous entities include events, nationalities, products and works of art.",
                    'Miscellaneous entities are used to identify and extract information about various types of entities from textual data that cannot be classified under any specific category.')
            span_positions = {"{};{}".format(start_positions[i], end_positions[i]):" ".join(words[start_positions[i]: end_positions[i] + 1]) for i in range(len(start_positions))}
            new_data = {
                'context':words,
                'end_position':end_positions,
                'entity_label':label,
                'impossible':data['impossible'],
                'qas_id':data['qas_id'],
                'query':query,
                'span_position':span_positions,
                'start_position': start_positions,
            }
            # if label == "ORG":
            new_datas.append(new_data)
        self.all_data = new_datas

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
        tokenizer = self.tokenizer

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        sequence_added_tokens = (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
            if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
            else tokenizer.model_max_length - tokenizer.max_len_single_sentence
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

        truncated_query = tokenizer.encode(
            query, add_special_tokens=False, truncation=True, max_length=self.max_query_length
        )
        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            truncated_query,
            all_doc_tokens,
            truncation=truncation,
            padding=padding_strategy,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )
        tokens = encoded_dict['input_ids']
        type_ids = encoded_dict['token_type_ids']
        attn_mask = encoded_dict['attention_mask']

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. special tokens
        doc_offset = len(truncated_query) + sequence_added_tokens
        new_start_positions = [x + doc_offset for x in tok_start_positions if (x + doc_offset) < self.max_length - 1]
        new_end_positions = [x + doc_offset if (x + doc_offset) < self.max_length - 1 else self.max_length - 2 for x in
                             tok_end_positions]
        new_end_positions = new_end_positions[:len(new_start_positions)]

        label_mask = [0] * doc_offset + [1] * (len(tokens) - doc_offset - 1) + [0]

        assert all(label_mask[p] != 0 for p in new_start_positions)
        assert all(label_mask[p] != 0 for p in new_end_positions)

        assert len(label_mask) == len(tokens)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(attn_mask),
            torch.LongTensor(type_ids),
            torch.LongTensor(label_mask),
            match_labels,
        ]