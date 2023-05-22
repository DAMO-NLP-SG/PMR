import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
from utils import SquadV1Processor
import random
import os
from tqdm import tqdm
logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet", 'xlmroberta'}
random.seed(42)

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

class QADataset(Dataset):
    def __init__(self, features, tokenizer: None, data_name: None, max_length: int = 512, max_query_length=64):
        self.all_data = features
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length

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
        label_mask = [0] * (data.doc_offset) + [1] * (seq_len - data.doc_offset - 1) + [0]

        assert all(label_mask[p] != 0 for p in data.start_positions)
        assert all(label_mask[p] != 0 for p in data.end_positions)
        assert len(label_mask) == seq_len

        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        if data.start_positions != [] and data.end_positions != []:
            for start, end in zip(data.start_positions, data.end_positions):
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

        return [
            torch.LongTensor(data.input_ids),
            torch.LongTensor(data.attention_mask),
            torch.LongTensor(data.token_type_ids),
            torch.LongTensor(label_mask),
            match_labels,
            torch.LongTensor([item]),
        ]

def cache_QAexamples(args, tokenizer, dataset_name, split, lang='en'):
    num_qa = 0
    doc_stride = args.doc_stride
    processor = SquadV1Processor()
    if split == "train" and dataset_name == "TyDiQA":
        input_data = processor.get_examples(os.path.join('./Data', 'tydiqa', 'tydiqa-goldp-v1.1-train'),
                                            "tydiqa.goldp.en.train.json")
    elif split == "dev" and dataset_name == "TyDiQA":
        input_data = processor.get_examples(os.path.join('./Data', 'tydiqa', 'tydiqa-goldp-v1.1-dev'),
                                            "tydiqa.goldp." + lang + ".dev.json")
    elif split == "train" and dataset_name == "SQuAD":
        input_data = processor.get_examples(os.path.join('./Data', 'squad'), "train-v1.1.json")
    elif split == "dev" and dataset_name == "SQuAD":
        input_data = processor.get_examples(os.path.join('./Data', 'squad'), "dev-v1.1.json")
    elif split == "dev" and dataset_name == "XQuAD":
        input_data = processor.get_examples(os.path.join('./Data', 'xquad'), "xquad." + lang + ".json")
    elif split == "dev" and dataset_name == "MLQA":
        input_data = processor.get_examples(os.path.join('./Data', 'mlqa/test'), 'test-context-' + lang + "-question-" + lang + ".json")
    features = []
    for i_t, item in enumerate(tqdm(input_data, desc="creating {} features from {}".format(split, dataset_name))):
        context = item.doc_tokens
        context_str = item.context_text
        char_to_word_offset = item.char_to_word_offset
        # convert context to index
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token_item) in enumerate(context):
            token = token_item[0]
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

        num_qa += 1
        qid = item.qas_id
        question = item.question_text
        answers = item.answers


        start_positions = [char_to_word_offset[x['answer_start']] for x in answers]
        end_positions = [char_to_word_offset[min(x['answer_start'] + len(x['text']) - 1, len(char_to_word_offset) - 1)] for x in answers]
        # query = 'Highlight the parts (if any) related to "{}". Details: {}'.format('the question', question)
        query = question
        truncated_query = tokenizer.encode(
            query, add_special_tokens=False, truncation=True, max_length=args.max_query_length
        )

        tok_start_positions = [orig_to_tok_index[x] for x in start_positions]
        tok_end_positions = []
        for x in end_positions:
            if x < len(context) - 1:
                tok_end_positions.append(orig_to_tok_index[x + 1] - 1)
            else:
                tok_end_positions.append(len(all_doc_tokens) - 1)
        se_improved = []
        for i_tok in range(len(tok_start_positions)):

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_positions[i_tok], tok_end_positions[i_tok], tokenizer, answers[i_tok]['text'], lang=lang
            )
            se_improved.append((tok_start_position, tok_end_position))
        tok_start_positions = [x[0] for x in se_improved]
        tok_end_positions = [x[1] for x in se_improved]
        spans = []
        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):
            sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
            doc_one = span_doc_tokens
            tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
            sequence_added_tokens = (
                tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
                if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
                else tokenizer.model_max_length - tokenizer.max_len_single_sentence
            )
            truncation = TruncationStrategy.ONLY_SECOND.value
            padding_strategy = "do_not_pad"

            encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                truncated_query,
                doc_one,
                truncation=truncation,
                padding=padding_strategy,
                max_length=args.max_seq_length,
                return_overflowing_tokens=True,
                return_token_type_ids=True,
                stride=args.max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            )
            doc_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                args.max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )  # remaining vs max-length limitation
            token_to_orig_map = {}
            for i in range(doc_len):
                index = len(truncated_query) + sequence_added_tokens + i  # 1 denotes cls
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            doc_start = len(spans) * doc_stride
            doc_end = doc_start + doc_len - 1
            tokens = encoded_dict['input_ids']
            type_ids = encoded_dict['token_type_ids']
            attn_mask = encoded_dict['attention_mask']
            # find new start_positions/end_positions, considering
            # 1. we add cls token at the beginning

            doc_offset = len(truncated_query) + sequence_added_tokens
            new_start_positions = [(ix, x - doc_start + doc_offset) for ix, x
                                   in enumerate(tok_start_positions) if x >= doc_start]
            new_end_positions = [(ix, x - doc_start + doc_offset) for ix, x
                                 in enumerate(tok_end_positions) if x <= doc_end]
            common_ixs = list(
                set([x[0] for x in new_start_positions]).intersection(set([x[0] for x in new_end_positions])))
            new_start_positions = [x[1] for x in new_start_positions if x[0] in common_ixs]
            new_end_positions = [x[1] for x in new_end_positions if x[0] in common_ixs]
            assert len(new_start_positions) == len(new_end_positions)
            span_doc_tokens = encoded_dict['overflowing_tokens']
            spans.append(encoded_dict)
            if split == 'train':
                feature_one = QAFeatures(
                        tokens,
                        attn_mask,
                        type_ids,
                        new_start_positions,
                        new_end_positions,
                        doc_offset,
                        qid,
                    )
            else:
                feature_one = QAFeatures(
                    tokens,
                    attn_mask,
                    type_ids,
                    new_start_positions,
                    new_end_positions,
                    doc_offset,
                    qid,
                    token_to_orig_map=token_to_orig_map,
                    context=context,
                    context_str=context_str,
                )
            features.append(feature_one)
            if len(span_doc_tokens) == 0:
                break
    logger.warning("finish creating {} features on {} questions".format(len(features), num_qa))
    return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text, lang):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text)).replace('\u0120', '').replace('▁','').strip() #handle RoBERTa/XLMR BPE tokenizer
    if lang == 'zh':
        tok_answer_text = tok_answer_text.replace(" ",'')
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)]).replace('\u0120', '').replace('▁','').strip()
            if lang == 'zh':
                text_span = text_span.replace(" ", "")
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)