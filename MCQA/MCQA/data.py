#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
import random
logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}
random.seed(42)

class MCQADataset(Dataset):
    def __init__(self, dataset, data_path, tokenizer: None, max_length: int = 512, max_query_length=64):
        self.data_name = data_path
        self.get_data(data_path, dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length
    def get_data(self, data_path, dataset):
        if data_path.startswith("DREAM"):
            self.all_data = []
            label_dict = {0:"A",1:'B',2:"C"}
            for raw_data_one in dataset:
                passage = " ".join(raw_data_one[0])
                questions = raw_data_one[1]
                for question_item in questions:
                    question = question_item["question"]
                    choice = question_item["choice"]
                    label = choice.index(question_item['answer'])
                    data_one = {
                        'passage': passage,
                        'question': question,
                        'choice': [label_dict[i_x] + ": " + x for i_x,x in enumerate(choice)],
                        'label': label
                    }
                    self.all_data.append(data_one)
        elif data_path == "MCTest":
            self.all_data = []
            label_dict = {0: "A", 1: 'B', 2: "C", 3: 'D'}
            for raw_data_one in dataset:
                input_data = raw_data_one[0]
                assert len(input_data) == 23
                labels = raw_data_one[1]
                passage = input_data[2]
                for i in range(4): # 4 questions for each passage
                    question = input_data[3 + i * 5].replace('multiple:', "").replace('one:', "")
                    choice = input_data[4 + i * 5: 4 + i * 5 + 4]
                    label = ord(labels[i]) - ord('A')
                    assert 0 <= label < 4
                    data_one = {
                        'passage': passage,
                        'question': question,
                        'choice': [label_dict[i_x] + ": " + x for i_x, x in enumerate(choice)],
                        'label': label
                    }
                    self.all_data.append(data_one)
        elif data_path.startswith("RACE"):
            self.all_data = []
            label_dict = {0: "A", 1: 'B', 2: "C", 3: 'D'}
            for raw_data_one in dataset:
                passage = raw_data_one[0][0]
                questions = raw_data_one[1]
                for question_item in questions: # 4 questions for each passage
                    question = question_item['question']

                    choice = question_item['choice']
                    assert len(choice) == 4
                    label = choice.index(question_item['answer'])
                    assert 0 <= label < 4
                    data_one = {
                        'passage': passage,
                        'question': question,
                        'choice': [label_dict[i_x] + ": " + x for i_x,x in enumerate(choice)],
                        'label': label
                    }
                    self.all_data.append(data_one)
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
        passage = data['passage']
        question = data['question']
        choice = data['choice']
        gold_label = data['label']
        # query = 'Highlight the parts (if any) related to "{}". Details: {}'.format('the question', question)
        tokenizer = self.tokenizer
        tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        sequence_added_tokens = (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
            if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
            else tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            question_tokens = tokenizer.tokenize(question, add_prefix_space=True)
        elif tokenizer.__class__.__name__ in [
            'BertTokenizer'
        ]:
            question_tokens = tokenizer.tokenize(question)
        elif tokenizer.__class__.__name__ in [
            'BertWordPieceTokenizer'
        ]:
            question_tokens = tokenizer.encode(question, add_special_tokens=False).tokens
        else:
            question_tokens = tokenizer.tokenize(question)

        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            passage_token = tokenizer.tokenize(passage, add_prefix_space=True)
        elif tokenizer.__class__.__name__ in [
            'BertTokenizer'
        ]:
            passage_token = tokenizer.tokenize(passage)
        elif tokenizer.__class__.__name__ in [
            'BertWordPieceTokenizer'
        ]:
            passage_token = tokenizer.encode(passage, add_special_tokens=False).tokens
        else:
            passage_token = tokenizer.tokenize(passage)

        spans = []
        for (i_c, token) in enumerate(choice):
            all_choice_tokens = []
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                all_choice_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            elif tokenizer.__class__.__name__ in [
                'BertTokenizer'
            ]:
                all_choice_tokens = tokenizer.tokenize(token)
            elif tokenizer.__class__.__name__ in [
                'BertWordPieceTokenizer'
            ]:
                all_choice_tokens = tokenizer.encode(token, add_special_tokens=False).tokens
            else:
                all_choice_tokens = tokenizer.tokenize(token)


            truncated_query = tokenizer.encode(
                question_tokens + all_choice_tokens, add_special_tokens=False, truncation=True, max_length=self.max_query_length
            )

            truncation = TruncationStrategy.ONLY_SECOND.value
            padding_strategy = "do_not_pad"

            encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                truncated_query,
                passage_token,
                truncation=truncation,
                padding=padding_strategy,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_token_type_ids=True,
            )
            tokens = encoded_dict['input_ids']
            type_ids = encoded_dict['token_type_ids']
            attn_mask = encoded_dict['attention_mask']

            seq_len = len(tokens)
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            if i_c == gold_label:
                match_labels[0, 0] = 1
            label_mask = [1] + [0] * (len(tokens) - 1)


            assert len(label_mask) == len(tokens)
            span = [
            torch.LongTensor(tokens),
            torch.LongTensor(attn_mask),
            torch.LongTensor(type_ids),
            torch.LongTensor(label_mask),
            match_labels,
            torch.LongTensor([gold_label]),
            item,
            ]
            spans.append(span)

        return spans


