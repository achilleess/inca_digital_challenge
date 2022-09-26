import os.path as osp
import copy
import json
from collections import defaultdict
import re

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BartTokenizer
from transformers import DebertaV2Tokenizer
from transformers import GPT2Tokenizer


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


class BotDetectionDataset(Dataset):
    def __init__(self, df, fold_to_train, tokenizer, infer_params, mode='train'):

        assert mode in ['train', 'val']

        self.tokenizer = tokenizer
        self.infer_params = infer_params
        self.mode = mode

        df['fold'] = df['fold'].astype(int)

        df = df[df['fold'] != -1]

        if self.mode == 'train':
            self.df = df[df['fold'] != np.int64(fold_to_train)].copy()
        else:
            self.df = df[df['fold'] == np.int64(fold_to_train)].copy()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        res_dict = {'indexes': index}

        row = self.df.iloc[index]

        if 'agreement_rate' in row:
            target = row.agreement_rate
            res_dict['target'] = target

        text = row.Text
        text = remove_html(text)
        text = remove_emoji(text)
        text = remove_URL(text)

        tokenizer_output = self.tokenizer.encode_plus(
            text=text, **self.infer_params
        )

        res_dict['input_ids'] = tokenizer_output['input_ids'][0]
        res_dict['attention_mask'] = tokenizer_output['attention_mask'][0]
        if 'token_type_ids' in tokenizer_output:
            res_dict['token_type_ids'] = tokenizer_output['token_type_ids'][0]
        return res_dict


def build_tokenizer(config):
    tokenizer_config = copy.deepcopy(config.tokenizer)
    tok_type = tokenizer_config.pop('type')
    infer_params = tokenizer_config.pop('infer_params')
    
    tokenizer = eval(tok_type).from_pretrained(**tokenizer_config)
    return tokenizer, infer_params


def get_loaders(config, df, val_only=False):
    tokenizer, infer_params = build_tokenizer(config)

    train_dataset_config = copy.deepcopy(config.train_dataset)

    if not val_only:
        dataset_type = train_dataset_config.pop('type')
        train_dataset = eval(dataset_type)(
            df=df,
            fold_to_train=config.fold_to_train,
            tokenizer=tokenizer,
            infer_params=infer_params,
            **train_dataset_config
        )

    #for i in range(0, len(train_dataset), 1):
    #    train_dataset[i]
    #assert 0

    val_dataset_config = copy.deepcopy(config.val_dataset)

    dataset_type = val_dataset_config.pop('type')
    val_dataset = eval(dataset_type)(
        df=df,
        fold_to_train=config.fold_to_train,
        tokenizer=tokenizer,
        infer_params=infer_params,
        **val_dataset_config
    )

    if not val_only:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **config.train_dataloader
        )

    valid_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **config.val_dataloader
    )

    if not val_only:
        return train_loader, valid_loader
    else:
        return valid_loader