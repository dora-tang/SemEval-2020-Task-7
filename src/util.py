import argparse
import random
import time
import os
import math
import re
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import shutil

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from allennlp.modules.scalar_mix import ScalarMix
from transformers import *

from torchtext import data
from torchtext.vocab import Vectors
from torchtext.vocab import GloVe

# import _pickle as pkl

import logging as log
# print = log.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_log():
    log.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
    log.getLogger().setLevel(log.INFO)
    log.getLogger().handlers.clear()
    log.getLogger().addHandler(log.StreamHandler())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_transformers():
    MODELS = [
        (BertModel,       BertTokenizer,       'bert-base-uncased'),
        (BertModel,       BertTokenizer,       'bert-large-uncased'),
        # (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
        # (GPT2Model,       GPT2Tokenizer,       'gpt2'),
        # (CTRLModel,       CTRLTokenizer,       'ctrl'),
        # (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
        # (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
        # (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
        # (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
        (RobertaModel,    RobertaTokenizer,    'roberta-base',),
        (RobertaModel,    RobertaTokenizer,    'roberta-large',),
        (RobertaModel,    RobertaTokenizer,    'roberta-large-mnli',),
        # (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
    ]

    MODELS_DICT = {}
    for model_class, tokenizer_class, model_name in MODELS:
        MODELS_DICT[model_name] = [model_class, tokenizer_class]
    return MODELS_DICT


def read_csv_split(train_path, valid_path, final_valid_path, no_split=False):
    df_list = []
    for i in train_path:
        #train = pd.read_csv(train_path)
        df_list.append(pd.read_csv(i))
    train = pd.concat(df_list)
    valid = pd.read_csv(valid_path)
    final_valid = pd.read_csv(final_valid_path)

    if no_split:
        test = valid
    else:
        # split
        test = valid[-1000:]
        test = test.reset_index(drop=True)
        valid = valid[:-1000]
        valid = valid.reset_index(drop=True)

    return train, valid, test, final_valid



def tokenize_and_cut(sentence, tokenizer):
    # for pretrained transformer
    sep_token = tokenizer.sep_token
    word = re.findall('<(.*)/>', sentence)[0]
    sentence = re.sub('<.*/>', f'{sep_token} {word} {sep_token}', sentence)
    tokens = tokenizer.tokenize(sentence)
    # tokens = tokens[:max_input_length-2]
    # tokens = tokens[:max_input_length]
    return tokens


def load_word_embedding(TEXT, train_data):
    log.info('')
    # glove
    # TEXT.build_vocab(train_data, vectors="fasttext.en.300d", unk_init=torch.Tensor.normal_)
    TEXT.build_vocab(train_data, vectors="glove.840B.300d", unk_init=torch.Tensor.normal_)
    # TEXT.build_vocab(train_data)
    log.info(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    log.info(TEXT.vocab.freqs.most_common(20))
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]  # 0
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # 1
    # pkl.dump(TEXT, open("emb", "wb"))
    log.info('')
    pretrained_embeddings = TEXT.vocab.vectors
    EMBEDDING_DIM = pretrained_embeddings.shape[-1]
    pretrained_embeddings[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    pretrained_embeddings[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    # pretrained_embeddings[UNK_IDX] = pretrained_embeddings[2:].mean(0)
    embedding = nn.Embedding(*pretrained_embeddings.shape)
    embedding.weight.data.copy_(pretrained_embeddings)

    return TEXT, embedding


class RMSE():
    def __init__(self):
        self.n_instance = 0
        self.sum_of_square = 0

    def accumulate(self, sse, num):
        self.sum_of_square += sse
        self.n_instance += num

    def calculate(self, clear=True):
        rmse = np.sqrt(self.sum_of_square / self.n_instance)
        if clear:
            self._clear()
        return rmse

    def _clear(self):
        self.n_instance = 0
        self.sum_of_square = 0


class Classifier(nn.Module):
    def __init__(self, d_inp, n_classes, cls_type="log_reg", dropout=0.2, d_hid=512):
        super().__init__()

        # logistic regression
        if cls_type == "log_reg":
            classifier = nn.Linear(d_inp, n_classes)
        # mlp
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, n_classes),
            )
        self.classifier = classifier

    def forward(self, seq_emb):
        logits = self.classifier(seq_emb)
        return logits


class Pooler(nn.Module):
    """ Do pooling, possibly with a projection beforehand """

    def __init__(self, project=True, d_inp=512, d_proj=512, pool_type="max"):
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

    def forward(self, sequence, mask=None):
        """
        sequence: (bsz, T, d_inp)
        mask: nopad_mask (bsz, T) or (bsz, T, 1)
        """

        # sequence is (bsz, d_inp), no need to pool
        if len(sequence.shape) == 2:
            return sequence

        # no pad in sequence
        if mask is None:
            mask = torch.ones(sequence.shape[:2], device=device)

        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)  # (bsz, T, 1)
        pad_mask = (mask == 0)
        proj_seq = self.project(sequence)  # (bsz, T, d_proj) or (bsz, T, d_inp)

        if self.pool_type == "max":
            proj_seq = proj_seq.masked_fill(pad_mask, -float("inf"))
            seq_emb = proj_seq.max(dim=1)[0]

        elif self.pool_type == "mean":
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1).float()

        return seq_emb
