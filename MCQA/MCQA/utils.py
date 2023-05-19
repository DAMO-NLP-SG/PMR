import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)

def collate_to_max_length_roberta(batch):
    """
    adapted form https://github.com/ShannonAI/mrc-for-flat-nested-ner
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(y[0].shape[0] for x in batch for y in x)
    choice_size = len(batch[0])
    output = []

    for field_idx in range(4):
        if field_idx == 0:
            pad_output = torch.full([batch_size, choice_size, max_length], 1, dtype=batch[0][0][field_idx].dtype)
        else:
            pad_output = torch.full([batch_size, choice_size, max_length], 0, dtype=batch[0][0][field_idx].dtype)
        for batch_idx in range(len(batch)):
            for choice_idx in range(len(batch[batch_idx])):
                data = batch[batch_idx][choice_idx][field_idx]
                pad_output[batch_idx][choice_idx][: data.shape[0]] = data
        output.append(pad_output)

    data_itemid = []
    pad_labels = torch.zeros([batch_size], dtype=torch.long)
    pad_match_labels = torch.zeros([batch_size, choice_size, max_length, max_length], dtype=torch.long)
    for batch_idx in range(len(batch)):
        pad_labels[batch_idx] = batch[batch_idx][0][5]
        data_itemid.append(batch[batch_idx][0][6])
        for choice_idx in range(len(batch[batch_idx])):
            data = batch[batch_idx][choice_idx][4]
            pad_match_labels[batch_idx, choice_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)
    output.append(pad_labels)
    output.append(data_itemid)
    return output

def get_preds_ans(match_logits, item_ids, dataset):
    batch_size, choice_size, seq_len, seq_len = match_logits.size()
    result_dict = {}
    for ib in range(batch_size):
        match_logits_one = match_logits[ib]
        item_one = int(item_ids[ib])
        dataset_one = dataset.all_data[item_one]
        label_logits = match_logits_one[:, 0, 0]
        pred = int(torch.argmax(label_logits))
        label = dataset_one['label']
        result_dict[item_one] = [pred, label]

    return result_dict





