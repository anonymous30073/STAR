import json
import os
import pickle
import random
from builtins import NotImplementedError
from collections import Counter, defaultdict, OrderedDict
from os.path import join, exists

import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np

pd.set_option('display.max_columns', 50)
INTERNAL_USER_ID_FIELD = "internal_user_id"
INTERNAL_ITEM_ID_FIELD = "internal_item_id"


def tokenize_function(examples, tokenizer, field, max_length, padding):
    result = tokenizer(
        examples[field],
        truncation=True,
        max_length=max_length,
        padding=padding
    )
    examples['input_ids'] = result['input_ids']
    examples['attention_mask'] = result['attention_mask']
    return examples


def load_data(config, pretrained_model=None, joint=False):

    # read data files
    datasets, user_info, item_info, val_negs, test_negs, item_mapping, user_mapping, test_label_list = load_split_dataset(
        config)

    # tokenize textual user/item data
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
    padding_token = tokenizer.pad_token_id
    return_padding_token = tokenizer.pad_token_id
    user_info = user_info.map(tokenize_function, batched=True,
                              fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                         "max_length": config["user_chunk_size"],
                                         "padding": False
                                         })
    user_info = user_info.remove_columns(['text'])
    item_info = item_info.map(tokenize_function, batched=True,
                              fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                         "max_length": config["item_chunk_size"],
                                         "padding": False
                                         })
    item_info = item_info.remove_columns(['text'])

    # collect seen items
    user_used_items = get_user_used_items(datasets)
    cur_used_items = user_used_items['train'].copy()

    # collate functions for dataloaders
    train_collate_fn = CollateNegSamplesRandomOpt(config['training_neg_samples'],
                                                  cur_used_items, user_info,
                                                  item_info, padding_token=padding_token, joint=joint)
    valid_collate_fn = CollateOriginalDataPadVal(user_info, item_info, val_negs, padding_token=padding_token,
                                                 joint=joint)
    test_collate_fn = CollateOriginalDataPadVal(user_info, item_info, test_negs, padding_token=padding_token,
                                                joint=joint)

    # train/val/test dataloaders
    train_dataloader = DataLoader(datasets['train'],
                                  batch_size=config['train_batch_size'],
                                  shuffle=True,
                                  collate_fn=train_collate_fn,
                                  num_workers=config['dataloader_num_workers']
                                  )
    validation_dataloader = DataLoader(datasets['validation'],
                                       batch_size=config['eval_batch_size'],
                                       collate_fn=valid_collate_fn,
                                       num_workers=config['dataloader_num_workers'])
    test_dataloader = DataLoader(datasets['test'],
                                 batch_size=config['eval_batch_size'],
                                 collate_fn=test_collate_fn,
                                 num_workers=config['dataloader_num_workers'])

    return train_dataloader, validation_dataloader, test_dataloader, user_info, item_info, \
        config['relevance_level'] if 'relevance_level' in config else None, return_padding_token, \
        item_mapping, user_mapping, test_label_list


# collate for train + parent class for eval
class CollateNegSamplesRandomOpt(object):
    def __init__(self, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None, joint=False):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        self.all_items = list(set(item_info[INTERNAL_ITEM_ID_FIELD]))
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token
        self.joint = joint

    def sample(self, batch_df):
        user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
        samples = []
        for user_id in sorted(user_counter.keys()):
            num_pos = user_counter[user_id]
            max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
            if max_num_user_neg_samples < num_pos * self.num_neg_samples:
                print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
                      f"but all_items are {len(self.all_items)}")
                pass
            user_samples = set()
            try_cnt = -1
            num_user_neg_samples = max_num_user_neg_samples
            while True:
                if try_cnt == 100:
                    print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
                          f"{user_id}. We instead have {len(user_samples)} samples.")
                    break
                current_samples = set(random.sample(self.all_items, num_user_neg_samples))
                current_samples -= user_samples
                cur_used_samples = self.used_items[user_id].intersection(current_samples)
                current_samples = current_samples - cur_used_samples
                user_samples = user_samples.union(current_samples)
                num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
                if len(user_samples) < max_num_user_neg_samples:
                    if num_user_neg_samples < len(user_samples):
                        num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
                    try_cnt += 1
                else:
                    if len(user_samples) > max_num_user_neg_samples:
                        user_samples = set(list(user_samples)[:max_num_user_neg_samples])
                    break
            samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
                            for sampled_item_id in user_samples])
        return samples

    def prepare_text_pad(self, batch_df):
        ret = {}
        temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][
            ['input_ids', 'attention_mask']] \
            .reset_index().drop(columns=['index'])
        temp_user = pd.concat([batch_df, temp_user], axis=1)
        temp_user = temp_user.rename(columns={"input_ids": "user_input_ids",
                                              "attention_mask": "user_attention_mask"})
        temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][
            ['input_ids', 'attention_mask']] \
            .reset_index().drop(columns=['index'])
        temp_item = pd.concat([batch_df, temp_item], axis=1)
        temp_item = temp_item.rename(columns={"input_ids": "item_input_ids",
                                              "attention_mask": "item_attention_mask"})
        if "ref_item" in temp_item:
            temp = pd.merge(temp_user[[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label', 'user_input_ids',
                                       'user_attention_mask', "ref_item"]],
                            temp_item[[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label', 'item_input_ids',
                                       'item_attention_mask', "ref_item"]],
                            on=[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label', "ref_item"])
            temp = temp.drop(columns=["ref_item"])
        else:
            temp = pd.merge(temp_user[[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label', 'user_input_ids',
                                       'user_attention_mask']],
                            temp_item[[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label', 'item_input_ids',
                                       'item_attention_mask']],
                            on=[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label'])
        cols_to_pad = ["user_input_ids", "user_attention_mask", "item_input_ids", "item_attention_mask"]
        if self.joint:
            temp["input_ids"] = temp["user_input_ids"].apply(lambda x: x.tolist()) + temp["item_input_ids"].apply(
                lambda x: x.tolist()[1:])
            temp["attention_mask"] = temp["user_attention_mask"].apply(lambda x: x.tolist()) + temp[
                "item_attention_mask"].apply(lambda x: x.tolist()[1:])
            temp = temp.drop(columns=["user_input_ids", "user_attention_mask",
                                      "item_input_ids", "item_attention_mask"])
            cols_to_pad = ["attention_mask", "input_ids"]
        for col in cols_to_pad:
            ret[col] = pad_sequence([torch.tensor(t) for t in temp[col]], batch_first=True,
                                    padding_value=self.padding_token)
        for col in temp.columns:
            if col in ret:
                continue
            ret[col] = torch.tensor(temp[col]).unsqueeze(1)
        return ret

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch).sort_values(INTERNAL_USER_ID_FIELD)
        batch_size = len(batch_df)
        samples = pd.DataFrame(self.sample(batch_df)).sort_values(INTERNAL_USER_ID_FIELD)
        dummy_index = [[x for x in range(i, len(samples), batch_size)] for i in range(batch_size)]
        samples["dummy1"] = [x for l in dummy_index for x in l]
        samples = samples.sort_values("dummy1")
        batch_df = pd.concat([batch_df, samples]).reset_index().drop(columns=['index']).drop(columns=['dummy1'])
        ret = self.prepare_text_pad(batch_df)
        return ret

# validation/test collate
class CollateOriginalDataPadVal(CollateNegSamplesRandomOpt):
    def __init__(self, user_info, item_info, val_negs, padding_token=None, joint=False):
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token
        self.joint = joint
        self.val_negs = val_negs

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        batch_size = len(batch_df)
        samples = pd.DataFrame(self.val_negs[self.val_negs['idx'].isin(batch_df["idx"].tolist())])
        dummy_index = [[x for x in range(i, len(samples), batch_size)] for i in range(batch_size)]
        samples["dummy1"] = [x for l in dummy_index for x in l]
        samples = samples.sort_values("dummy1")
        batch_df = pd.concat([batch_df, samples]).reset_index().drop(columns=['index']).drop(columns=['dummy1'])
        ret = self.prepare_text_pad(batch_df)
        return ret


def get_user_used_items(datasets):
    used_items = {}
    for split in datasets.keys():
        used_items[split] = defaultdict(set)
        for user_iid, item_iid in zip(datasets[split][INTERNAL_USER_ID_FIELD], datasets[split][INTERNAL_ITEM_ID_FIELD]):
            used_items[split][user_iid].add(item_iid)
    return used_items

# collate function for dumping bert_cross scores
class CollateUserItem(object):
    def __init__(self, padding_token, user):
        self.padding_token = padding_token
        self.user_input_ids = user["input_ids"]
        self.user_att_mask = user["attention_mask"]

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        ret = {}
        batch_df["input_ids"] = batch_df["input_ids"].apply(lambda x: self.user_input_ids + x[1:])
        batch_df["attention_mask"] = batch_df["attention_mask"].apply(lambda x: self.user_att_mask + x[1:])
        cols_to_pad = ["input_ids", "attention_mask"]
        for col in cols_to_pad:
            ret[col] = pad_sequence([torch.tensor(t) for t in batch_df[col]], batch_first=True,
                                    padding_value=self.padding_token)
        for col in batch_df.columns:
            if col in ret:
                continue
            if col in ["user_id", "item_id"]:
                ret[col] = batch_df[col]
                continue
        return ret

# loading datasets
def load_split_dataset(config):
    num_negs = int(config["test_neg_sampling_strategy"].split("_")[-1])
    user_text_file_name = config['user_text_file_name']
    item_text_file_name = config['item_text_file_name']

    # read user file
    keep_fields = ["user_id"]
    user_info = pd.read_csv(join(config['dataset_path'], "users.csv"), usecols=keep_fields, dtype=str)
    user_info = user_info.sort_values("user_id").reset_index(
        drop=True)
    user_info[INTERNAL_USER_ID_FIELD] = np.arange(0, user_info.shape[0])
    if len(user_info["user_id"]) != len(set(user_info["user_id"])):
        raise ValueError("problem in users.csv file")
    if user_text_file_name is not None:
        up = pd.read_csv(join(config['dataset_path'], f"users_profile_{user_text_file_name}.csv"), dtype=str)
        up = up.fillna('')
        if len(up["user_id"]) != len(set(user_info["user_id"])):
            raise ValueError(f"problem in users_profile_{user_text_file_name}.csv file")
        user_info = pd.merge(user_info, up, on="user_id")
        user_info['text'] = user_info['text'].apply(lambda x: x.replace("<end of review>", ""))
        if not config['case_sensitive']:
            user_info['text'] = user_info['text'].apply(str.lower)
        if config['normalize_negation']:
            user_info['text'] = user_info['text'].replace("n\'t", " not", regex=True)

    # read item file
    keep_fields = ["item_id"]
    item_info = pd.read_csv(join(config['dataset_path'], "items.csv"), usecols=keep_fields, low_memory=False, dtype=str,
                            converters={'category': pd.eval})
    item_info = item_info.sort_values("item_id").reset_index(
        drop=True)  # this is crucial, as the precomputing is done with internal ids
    item_info[INTERNAL_ITEM_ID_FIELD] = np.arange(0, item_info.shape[0])
    if len(item_info["item_id"]) != len(set(item_info["item_id"])):
        raise ValueError("problem in items.csv file")
    print(f"num items = {len(item_info[INTERNAL_ITEM_ID_FIELD])}")
    item_info = item_info.fillna('')
    if item_text_file_name is not None:
        ip = pd.read_csv(join(config['dataset_path'], f"item_profile_{item_text_file_name}.csv"), dtype=str)
        ip = ip.fillna('')
        if len(ip["item_id"]) != len(set(item_info["item_id"])):
            raise ValueError(f"problem in item_profile_{item_text_file_name}.csv file")
        item_info = pd.merge(item_info, ip, on="item_id")
        item_info['text'] = item_info['text'].apply(
            lambda x: x.replace("<end of review>", "").replace("&", "").replace("Books, ", ""))
        if not config['case_sensitive']:
            item_info['text'] = item_info['text'].apply(str.lower)
        if config['normalize_negation']:
            item_info['text'] = item_info['text'].replace("n\'t", " not", regex=True)

    if 'item.category' in item_info.columns:
        item_info['item.category'] = item_info['item.category'].apply(
            lambda x: ", ".join(x[1:-1].split(",")).replace("'", "").replace('"', "").replace("  ", " ")
            .replace("[", "").replace("]", "").strip())
    if "category" in item_info.columns:
        item_info["category"] = item_info["category"].apply(
            lambda x: " ".join([y.replace("&", "").lower() for y in x if y != "Books"]) + " ")
        item_info["text"] = item_info["category"] + item_info["text"]
        item_info.drop(columns=["category"])

    # user item amazon ids to internal ids
    item_mapping = dict()
    user_mapping = dict()
    for i, j in zip(item_info["item_id"], item_info[INTERNAL_ITEM_ID_FIELD]):
        item_mapping[i] = j
    for i, j in zip(user_info["user_id"], user_info[INTERNAL_USER_ID_FIELD]):
        user_mapping[i] = j

    # read train/val/test files
    sp_files = {"train": join(config['dataset_path'], "train.csv"),
                "validation": join(config['dataset_path'], f'{config["alternative_pos_validation_file"]}.csv' if (
                        "alternative_pos_validation_file" in config and config[
                    "alternative_pos_validation_file"] != "") else "validation.csv"),
                "test": join(config['dataset_path'], f'{config["alternative_pos_test_file"]}.csv' if (
                        "alternative_pos_test_file" in config and config[
                    "alternative_pos_test_file"] != "") else "test.csv")}
    split_datasets = {}
    for sp, file in sp_files.items():
        df = pd.read_csv(file, usecols=["user_id", "item_id", "rating"], dtype=str)

        df['rating'] = df['rating'].fillna(-1)
        df['rating'] = df['rating'].astype(float).astype(int)
        df['label'] = np.ones(df.shape[0])

        df = df.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        df = df.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")

        df = df.drop(columns=["user_id", "item_id"])
        split_datasets[sp] = df

    test_label_list = dict()
    test_label_list["test"] = \
        split_datasets["test"].groupby(INTERNAL_USER_ID_FIELD)[INTERNAL_ITEM_ID_FIELD].apply(list).reset_index(
            name="tmp").sort_values(INTERNAL_USER_ID_FIELD)["tmp"].values.tolist()
    test_label_list["train"] = \
        split_datasets["train"].groupby(INTERNAL_USER_ID_FIELD)[INTERNAL_ITEM_ID_FIELD].apply(list).reset_index(
            name="tmp").sort_values(INTERNAL_USER_ID_FIELD)["tmp"].values.tolist()
    test_label_list["val"] = \
        split_datasets["validation"].groupby(INTERNAL_USER_ID_FIELD)[INTERNAL_ITEM_ID_FIELD].apply(list).reset_index(
            name="tmp").sort_values(INTERNAL_USER_ID_FIELD)["tmp"].values.tolist()

    # validation negs
    fname = config['validation_neg_sampling_strategy'][2:]
    negs = pd.read_csv(join(config['dataset_path'], fname + ".csv"), dtype=str)
    negs['label'] = 0
    negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
    negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
    negs = negs.drop(columns=["user_id", "item_id"])
    if "ref_item" in negs.columns:
        negs = negs.drop(columns=["ref_item"])
        print(f"number of test samples pos: {len(split_datasets['test'])} - neg (before drop dup): {len(negs)}")
        negs = negs.drop_duplicates()
    negs = negs.sort_values(INTERNAL_USER_ID_FIELD)
    split_datasets['validation'] = split_datasets['validation'].sort_values(INTERNAL_USER_ID_FIELD)
    if set(negs[INTERNAL_USER_ID_FIELD]) != set(split_datasets['validation'][INTERNAL_USER_ID_FIELD]):
        raise ValueError("user ids differ in validation file and validation neg file")

    split_datasets['validation'] = split_datasets['validation'].sort_values(INTERNAL_USER_ID_FIELD).reset_index().drop(
        columns=['index'])
    split_datasets['validation']["idx"] = range(len(split_datasets['validation']))
    dummy_column = [[x for _ in range(100)] for x in range(len(split_datasets['validation']))]
    val_negs = negs
    val_negs["idx"] = [x for l in dummy_column for x in l]

    # test negs
    fname = config['test_neg_sampling_strategy'][2:]
    negs = pd.read_csv(join(config['dataset_path'], fname + ".csv"), dtype=str)
    negs['label'] = 0
    negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
    negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
    negs = negs.drop(columns=["user_id", "item_id"])
    if "ref_item" in negs.columns and False:
        negs = negs.drop(columns=["ref_item"])
        print(f"number of test samples pos: {len(split_datasets['test'])} - neg (before drop dup): {len(negs)}")
        negs = negs.drop_duplicates()
    negs = negs.sort_values(INTERNAL_USER_ID_FIELD)
    split_datasets['test'] = split_datasets['test'].sort_values(INTERNAL_USER_ID_FIELD)

    print(f"number of test samples pos: {len(split_datasets['test'])} - neg: {len(negs)}")
    if set(negs[INTERNAL_USER_ID_FIELD]) != set(split_datasets['test'][INTERNAL_USER_ID_FIELD]):
        raise ValueError("user ids differ in test file and test neg file")
    if 'rating' in split_datasets['test']:
        split_datasets['test']['rating'] = split_datasets['test']['rating'].fillna(0)

    split_datasets['test'] = split_datasets['test'].sort_values(
        INTERNAL_USER_ID_FIELD).reset_index().drop(columns=['index'])
    split_datasets['test']["idx"] = range(len(split_datasets['test']))
    dummy_column = [[x for _ in range(num_negs)] for x in range(len(split_datasets['test']))]
    test_negs = negs
    test_negs["idx"] = [x for l in dummy_column for x in l]

    for split in split_datasets.keys():
        split_datasets[split] = Dataset.from_pandas(split_datasets[split], preserve_index=False)

    return DatasetDict(split_datasets), Dataset.from_pandas(user_info, preserve_index=False), \
        Dataset.from_pandas(item_info,
                            preserve_index=False), val_negs, test_negs, item_mapping, user_mapping, test_label_list
