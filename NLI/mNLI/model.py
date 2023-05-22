from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.modeling_outputs import (
    ModelOutput
)
import torch.nn.functional as F
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
from torch import nn
from transformers.utils import logging
from torch.nn import BCEWithLogitsLoss
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
logger = logging.get_logger(__name__)

class RoBERTa_PMR(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.span_transfer = MultiNonLinearProjection(config.hidden_size, config.hidden_size, config.hidden_dropout_prob,
                                                       intermediate_hidden_size=config.projection_intermediate_hidden_size)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_mask=None,
        match_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, choice_size, seq_length = input_ids.size()
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, seq_length)
        if match_labels is not None:
            label_mask = label_mask.view(-1, seq_length)
            match_labels = match_labels.view(-1, seq_length, seq_length)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # adapted from https://github.com/ShannonAI/mrc-for-flat-nested-ner
        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, hidden]
        span_intermediate = self.span_transfer(sequence_output)
        # [batch, seq_len, seq_len]
        span_logits = torch.matmul(span_intermediate, sequence_output.transpose(-1, -2))

        total_loss = None
        if match_labels is not None:
            match_loss = self.compute_loss(span_logits, match_labels, label_mask)
            # match_loss = self.compute_loss(span_logits, labels, label_mask, all_starts, all_ends)
            total_loss = match_loss
        span_logits = span_logits.view(batch_size, choice_size, seq_length, seq_length)
        if not return_dict:
            output = (span_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return PMROutput(
            loss=total_loss,
            span_logits=span_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def compute_loss(self, span_logits, match_labels, label_mask):
        batch_size, seq_len, seq_len = span_logits.size()

        match_label_row_mask = label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        loss_fct = BCEWithLogitsLoss(reduction="none")
        match_loss = loss_fct(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return match_loss

class XLMR_PMR(RoBERTa_PMR):
    """
    This class overrides [`RoBERTa_PMR`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig

class MultiNonLinearProjection(nn.Module):
    'copy from https://github.com/ShannonAI/mrc-for-flat-nested-ner'
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(MultiNonLinearProjection, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

@dataclass
class PMROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    match_loss: Optional[torch.FloatTensor] = None
    span_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]]= None