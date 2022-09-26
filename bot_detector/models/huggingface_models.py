import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers import BartForSequenceClassification
from transformers import DebertaV2ForSequenceClassification
from transformers import GPT2ForSequenceClassification

from .build_model import register_model


@register_model
class HuggingFaceModel(nn.Module):
    def __init__(self, model_type, model_name):
        super(HuggingFaceModel, self).__init__()

        self.model = eval(model_type).from_pretrained(model_name)
    
    def forward(self, container):
        model_inputs = dict(
          input_ids=container['input_ids'],
          attention_mask=container['attention_mask']
        )
        if 'token_type_ids' in container:
          model_inputs['token_type_ids'] = container['token_type_ids']
        model_output = self.model(**model_inputs)
        container['model_output'] = model_output.logits[:, 0]