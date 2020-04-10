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
import logging as log

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

# must import before torchtext.data, otherwise will override name
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW as huggingface_AdamW
from transformers import *

from torchtext import data
from torchtext.vocab import Vectors
from torchtext.vocab import GloVe

from allennlp.modules.scalar_mix import ScalarMix
from metrics import *


# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
# import _pickle as pkl
# print = log.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


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


def transformers_dict():
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


def init_transformers(model_name):
    MODELS_DICT = transformers_dict()
    model_class, tokenizer_class = MODELS_DICT[model_name]
    try:
        cache_dir = os.environ['HUGGINGFACE_TRANSFORMERS_CACHE']
    except KeyError as e:
        log.info('ERROR: environment variable HUGGINGFACE_TRANSFORMERS_CACHE not found'
                 + '\n\tmust define cache directory for huggingface transformers'
                 + '\n\tRUN THIS: export HUGGINGFACE_TRANSFORMERS_CACHE=/your/path'
                 + '\n\tyou should also append the line to ~/.bashrc (linux) or ~/.bash_profile (mac)'
                 + '\n')
        raise
    tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_dir)
    transformer = model_class.from_pretrained(
        model_name, cache_dir=cache_dir, output_hidden_states=True)
    return tokenizer, transformer


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


def args_default_path(args, default_data_dir):
    def default_path(path, default_path):
        if len(path) == 0:
            path = default_path
        return path

    args.train_path = default_path(args.train_path, os.path.join(default_data_dir, "train.csv"))
    args.train_extra_path = default_path(
        args.train_path, os.path.join(default_data_dir, "train_funlines.csv"))
    args.val_path = default_path(args.val_path, os.path.join(default_data_dir, "dev.csv"))
    args.test_with_label_path = default_path(
        args.test_with_label_path, os.path.join(default_data_dir, "test_with_label.csv"))
    args.test_without_label_path = default_path(
        args.test_without_label_path, os.path.join(default_data_dir, "test_without_label.csv"))
    return args


def get_task(args, get_dataset, text_field, mask_token):

    def read_task(
            train_path=None,
            val_path=None,
            test_with_label_path=None,
            test_without_label_path=None):
        task = {}
        log.info('')
        for split_name, path in[('train_data', 'train_path'), ('val_data', 'val_path'),
                                ('test_with_label_data', 'test_with_label_path'),
                                ('test_without_label_data', 'test_without_label_path')]:
            path = eval(path)
            if path is not None:
                log.info(f'read {split_name} from {path}')
                if split_name == 'train_data':
                    split = pd.concat([pd.read_csv(i) for i in path])
                else:
                    split = pd.read_csv(path)

                task[f'{split_name}'] = get_dataset(
                    csv_data=split, text_field=text_field, preprocess=args.model, mask_token=mask_token)

        for split_name in task.keys():
            length = len(task[f'{split_name}'])
            log.info(f'Number of examples in {split_name}: {length}')
        return task

    if args.train_extra:
        train_path_list = [args.train_path, args.train_extra_path]
    else:
        train_path_list = [args.train_path]
    val_path = test_with_label_path = test_without_label_path = None
    if args.do_train:
        val_path = args.val_path
    if args.do_eval:
        test_with_label_path = args.test_with_label_path
    if args.do_predict:
        test_without_label_path = args.test_without_label_path

    task = read_task(train_path=train_path_list, val_path=val_path, test_without_label_path=test_without_label_path,
                     test_with_label_path=test_with_label_path)
    return task


class Scheduler():
    def __init__(self, args, optimizer):
        if args.schedule == 'linear_schedule_with_warmup':
            #t_total = math.ceil(len(task['train_data']) / args.bsz) * args.epochs
            warmup_steps = 0
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.t_total)

        elif args.schedule == 'reduce_on_plateau':
            if args.task == 'task1':
                mode = "min"
            elif args.task == 'task2':
                mode = "max"
            scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=0.5,
                                          patience=3, min_lr=0, threshold=0.0001)

        elif args.schedule == 'none':
            scheduler = None

        self.schedule = args.schedule
        self.scheduler = scheduler

    def step(self, val_loss=None):
        if self.schedule == 'linear_schedule_with_warmup' and val_loss is None:
            self.scheduler.step()

        elif self.schedule == 'reduce_on_plateau' and val_loss is not None:
            self.scheduler.step(val_loss)
            log.info(
                "\t# validation passes without improvement: %d",
                self.scheduler.num_bad_epochs,
            )

    # def get_lr(self):
    #     return self.scheduler.get_lr()
    #
    #     # elif self.schedule == 'none':
    #     #     pass
