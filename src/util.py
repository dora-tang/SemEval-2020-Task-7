import argparse
import random
import time
import os
import math
import re
import sys
import copy
from tqdm import tqdm
import json
import shutil
import logging as log
from typing import Dict, Iterable, List, Sequence, Tuple, Union, Optional
from overrides import overrides

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

# must import before torchtext.data, otherwise will override name
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW as huggingface_AdamW
from transformers import *

from torchtext import data
from torchtext.vocab import Vectors, GloVe

from allennlp.modules.scalar_mix import ScalarMix

from metrics import *

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
# import _pickle as pkl
# print = log.info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def seed_all(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_log():
    log.basicConfig(
        format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
    )
    log.getLogger().setLevel(log.INFO)
    log.getLogger().handlers.clear()
    log.getLogger().addHandler(log.StreamHandler())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transformers_dict():
    MODELS = [
        (BertModel, BertTokenizer, "bert-base-uncased"),
        (BertModel, BertTokenizer, "bert-large-uncased"),
        # (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
        # (GPT2Model,       GPT2Tokenizer,       'gpt2'),
        # (CTRLModel,       CTRLTokenizer,       'ctrl'),
        # (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
        # (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
        # (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
        # (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
        (RobertaModel, RobertaTokenizer, "roberta-base",),
        (RobertaModel, RobertaTokenizer, "roberta-large",),
        (RobertaModel, RobertaTokenizer, "roberta-large-mnli",),
        # (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
    ]

    MODELS_DICT = {}
    for model_class, tokenizer_class, model_name in MODELS:
        MODELS_DICT[model_name] = [model_class, tokenizer_class]
    return MODELS_DICT


def init_transformers(model_name):
    """
    retrieve pretrained tokenizer and transformer model from name
    """
    MODELS_DICT = transformers_dict()
    model_class, tokenizer_class = MODELS_DICT[model_name]
    try:
        cache_dir = os.environ["HUGGINGFACE_TRANSFORMERS_CACHE"]
    except KeyError as e:
        log.info(
            "ERROR: environment variable HUGGINGFACE_TRANSFORMERS_CACHE not found"
            + "\n\tmust define cache directory for huggingface transformers"
            + "\n\tRUN THIS: export HUGGINGFACE_TRANSFORMERS_CACHE=/your/path"
            + "\n\tyou should also append the line to ~/.bashrc (linux) or ~/.bash_profile (mac)"
            + "\n"
        )
        raise
    tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_dir)
    transformer = model_class.from_pretrained(
        model_name, cache_dir=cache_dir, output_hidden_states=True
    )
    return tokenizer, transformer


def tokenize_and_cut(sentence, tokenizer):
    # for pretrained transformer
    sep_token = tokenizer.sep_token
    word = re.findall("<(.*)/>", sentence)[0]
    sentence = re.sub("<.*/>", f"{sep_token} {word} {sep_token}", sentence)
    tokens = tokenizer.tokenize(sentence)
    # tokens = tokens[:max_input_length-2]
    # tokens = tokens[:max_input_length]
    return tokens


def load_word_embedding(TEXT, train_data):
    log.info("")
    # glove
    # TEXT.build_vocab(train_data, vectors="fasttext.en.300d", unk_init=torch.Tensor.normal_)
    TEXT.build_vocab(
        train_data, vectors="glove.840B.300d", unk_init=torch.Tensor.normal_
    )
    # TEXT.build_vocab(train_data)
    log.info(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    log.info(TEXT.vocab.freqs.most_common(20))
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]  # 0
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # 1
    # pkl.dump(TEXT, open("emb", "wb"))
    log.info("")
    pretrained_embeddings = TEXT.vocab.vectors
    EMBEDDING_DIM = pretrained_embeddings.shape[-1]
    pretrained_embeddings[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    pretrained_embeddings[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    # pretrained_embeddings[UNK_IDX] = pretrained_embeddings[2:].mean(0)
    embedding = nn.Embedding(*pretrained_embeddings.shape)
    embedding.weight.data.copy_(pretrained_embeddings)

    return TEXT, embedding


def args_default_path(args, default_data_dir):
    def default_path(path, default_path):
        if len(path) == 0:
            path = default_path
        return path

    args.train_path = default_path(
        args.train_path, os.path.join(default_data_dir, "train.csv")
    )
    args.train_extra_path = default_path(
        args.train_path, os.path.join(default_data_dir, "train_funlines.csv")
    )
    args.val_path = default_path(
        args.val_path, os.path.join(default_data_dir, "dev.csv")
    )
    args.test_with_label_path = default_path(
        args.test_with_label_path, os.path.join(default_data_dir, "test_with_label.csv")
    )
    args.test_without_label_path = default_path(
        args.test_without_label_path,
        os.path.join(default_data_dir, "test_without_label.csv"),
    )
    return args
