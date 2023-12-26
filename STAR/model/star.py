import os.path
import time

import json
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

INTERNAL_USER_ID_FIELD = "internal_user_id"
INTERNAL_ITEM_ID_FIELD = "internal_item_id"

def calc_flops(batch_rep):
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

def dot_product(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(dim=-1)

class STAR(torch.nn.Module):
    def __init__(self, model_config, device):
        super(STAR, self).__init__()
        self.pretrained_name = model_config['pretrained_model']

        self.device = device

        self.bert = transformers.AutoModelForMaskedLM.from_pretrained(model_config['pretrained_model'])
        self.lm_dim = self.bert.config.vocab_size


    def forward(self, batch, mode = "train"):
        u_input_ids = batch['user_input_ids']
        u_att_mask = batch['user_attention_mask']
        i_input_ids = batch['item_input_ids']
        i_att_mask = batch['item_attention_mask']

        user_rep = torch.log(torch.relu(self.bert(input_ids=u_input_ids, attention_mask=u_att_mask).logits) + 1)
        user_rep = torch.max(user_rep * u_att_mask.unsqueeze(-1), dim=1)[0].squeeze(1)

        item_rep = torch.log(torch.relu(self.bert(input_ids=i_input_ids, attention_mask=i_att_mask).logits) + 1)
        item_rep = torch.max(item_rep * i_att_mask.unsqueeze(-1), dim=1)[0].squeeze(1)

        flops_user = calc_flops(user_rep)
        flops_item = calc_flops(item_rep)

        q_num = user_rep.size(0)
        d_num = item_rep.size(0)

        assert d_num % q_num == 0
        doc_group_size = d_num // q_num
        result = dot_product(
            user_rep.unsqueeze(1), item_rep.view(doc_group_size, q_num, -1).transpose(0, 1)
        ).T.contiguous()

        if mode == "train":
            pos_scores = result[0].view((-1,1))
            neg_scores = result[1:].view((-1,1))
            return result.view((-1,1)), flops_user, flops_item, pos_scores, neg_scores
        else:
            result = result.view((-1,1))
            return result, flops_user, flops_item, None, None
