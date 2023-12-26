import json
import os.path
import torch
import transformers
from torch.utils.data import DataLoader
from collections import defaultdict
from STAR.utils.data_loading_lsr import CollateUserItem
from tqdm import tqdm

class BertCross(torch.nn.Module):
    def __init__(self, model_config, device, dataset_config, exp_dir):
        super(BertCross, self).__init__()

        self.device = device
        self.config = transformers.AutoConfig.from_pretrained(model_config['pretrained_model'])
        self.config.num_labels = 1
        self.bert = transformers.AutoModelForSequenceClassification.from_pretrained(model_config['pretrained_model'], config = self.config)
        self.doc_section = dataset_config["training_neg_samples"] + 1
        self.exp_dir = exp_dir


    def forward(self, batch, mode = "train"):

        input_ids = batch['input_ids'].to("cuda")
        att_mask = batch['attention_mask'].to("cuda")

        result = self.bert(input_ids=input_ids, attention_mask=att_mask).logits

        if mode == "train":
            bs = input_ids.size()[0] // self.doc_section
            pos_scores = result[:bs]
            neg_scores = result[bs:]
            return result, 0, 0, pos_scores, neg_scores
        else:
            return result, 0, 0, None, None


    def prec_representations_for_distil(self, users, items, padding_token):
        print("computing representations")
        big_dict = defaultdict(dict)
        pbar = tqdm(enumerate(users), total=len(users))
        for i, u in pbar:
            collate_fn = CollateUserItem(padding_token=padding_token, user = u)
            uid = u["user_id"]
            dataloader = DataLoader(items, batch_size=128, collate_fn=collate_fn)
            for batch_idx, batch in enumerate(dataloader):
                scores, *_ = self.forward(batch, mode = "eval")
                scores = scores.detach()
                for iid, sc in zip(batch["item_id"], scores):
                    big_dict[uid][iid] = sc.item()
                del scores
                del batch
        print("finished computing representations")
        json.dump(dict(big_dict), open(os.path.join(self.exp_dir, "distil_scores.json"), "w"))

