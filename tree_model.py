# -*- coding: utf-8 -*-
# author: Victor H. Wirz
# discipline: UNIRIO-tin0145
# prof.: Pedro Moura

import os
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from collections import Counter
import pickle

TRAIN = 1
TEST = 2
DEV = 3

_UNK = '<unk>'


class SSTClient():
    def __init__(self, assets_dir="./assets/", sst_dir="./sst/"):
        self.assets_dir = assets_dir
        self.sst_dir = sst_dir

    def compile(self, tokens_file="SOStr.txt"):
        tokens_file = os.path.join(self.sst_dir, tokens_file)

        with open(tokens_file, 'r') as f:
            txt_raw = f.read()

        tokenizer = get_tokenizer(lambda p: p.split('|'), "basic_english")
        counter = Counter()

        for sentence in txt_raw.split("\n"):
            tokens = tokenizer(sentence)
            counter.update(tokens)

        lexis = Vocab(counter, min_freq=4)
        stoi = lexis.stoi

        if not os.path.exists(self.assets_dir + 'stoi.pkl'):
            pickle.dump(stoi, open(os.path.join(self.assets_dir, 'stoi.pkl'), 'wb'))


SSTClient().compile()
