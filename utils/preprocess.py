# -*- coding:utf-8 -*-

import collections
import logging
import operator
import os
from typing import List, Optional, Union, Dict, Any

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
# self-build
from utils.dataloader import read_data_file, get_list_subset, pad_sequences


class ClassificationProcessor(object):
    """
    Corpus Pre Processor class.
    Build tokenizer by word's frequency.
    """

    def __init__(self, multi_label=False, **kwargs):
        self.token2idx: Dict[str, int] = kwargs.get('token2idx', {})
        self.idx2token: Dict[int, str] = dict([(v, k) for (k, v) in self.token2idx.items()])

        self.token2count: Dict = {}

        self.label2idx: Dict[str, int] = kwargs.get('label2idx', {})
        self.idx2label: Dict[int, str] = dict([(v, k) for (k, v) in self.label2idx.items()])

        self.token_pad: str = kwargs.get('token_pad', '<PAD>')
        self.token_unk: str = kwargs.get('token_unk', '<UNK>')
        self.token_bos: str = kwargs.get('token_bos', '<BOS>')
        self.token_eos: str = kwargs.get('token_eos', '<EOS>')

        self.dataset_info: Dict[str, Any] = kwargs.get('dataset_info', {})

        self.add_bos_eos: bool = kwargs.get('add_bos_eos', False)

        self.sequence_length = kwargs.get('sequence_length', None)

        self.min_count = kwargs.get('min_count', 3)

        # classification
        self.multi_label = multi_label
        self.multi_label_binarizer: MultiLabelBinarizer = None

    def info(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'label2idx': self.label2idx,
                'token2idx': self.token2idx,
                'token_pad': self.token_pad,
                'token_unk': self.token_unk,
                'token_bos': self.token_bos,
                'token_eos': self.token_eos,
                'dataset_info': self.dataset_info,
                'add_bos_eos': self.add_bos_eos,
                'sequence_length': self.sequence_length
            },
            'module': self.__class__.__module__,
        }

    def analyze_corpus(self,
                       corpus: Union[List[List[str]]],
                       labels: Union[List[List[str]], List[str]],
                       force: bool = False):
        rec_len = sorted([len(seq) for seq in corpus])[int(0.95 * len(corpus))]
        self.dataset_info['RECOMMEND_LEN'] = rec_len

        if len(self.token2idx) == 0 or force:
            self._build_token_dict(corpus, self.min_count)
        if len(self.label2idx) == 0 or force:
            self._build_label_dict(labels)

    def _build_token_dict(self, corpus: List[List[str]], min_count: int = 3):
        """
        Build token index dictionary using corpus

        Args:
            corpus: List of tokenized sentences, like ``[['I', 'love', 'tf'], ...]``
            min_count:
        """
        token2idx = {
            self.token_pad: 0,
            self.token_unk: 1,
            self.token_bos: 2,
            self.token_eos: 3
        }

        token2count = {}
        for sentence in corpus:
            for token in sentence:
                count = token2count.get(token, 0)
                token2count[token] = count + 1

        # 按照词频降序排序
        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        token2count = collections.OrderedDict(sorted_token2count)

        for token, token_count in token2count.items():
            if token not in token2idx and token_count >= min_count:
                token2idx[token] = len(token2idx)

        self.token2idx = token2idx
        self.idx2token = dict([(value, key)
                               for key, value in self.token2idx.items()])
        logging.debug(f"build token2idx dict finished, contains {len(self.token2idx)} tokens.")
        self.dataset_info['token_count'] = len(self.token2idx)

    def _build_label_dict(self,
                          labels: List[str]):
        if self.multi_label:
            label_set = set()
            for i in labels:
                label_set = label_set.union(list(i))
        else:
            label_set = set(labels)
        self.label2idx = {}
        for idx, label in enumerate(sorted(label_set)):
            self.label2idx[label] = len(self.label2idx)

        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        self.dataset_info['label_count'] = len(self.label2idx)
        self.multi_label_binarizer = MultiLabelBinarizer(classes=list(self.label2idx.keys()))

    def process_x_dataset(self,
                          data: List[List[str]],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if max_len is None:
            max_len = self.sequence_length
        if subset is not None:
            target = get_list_subset(data, subset)
        else:
            target = data
        numerized_samples = self.numerize_token_sequences(target)

        return pad_sequences(numerized_samples, max_len, padding='post', truncating='post')

    def process_y_dataset(self,
                          data: List[str],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if subset is not None:
            target = get_list_subset(data, subset)
        else:
            target = data
        numerized_samples = self.numerize_label_sequences(target)
        return np.array(numerized_samples, dtype='int')

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):
        result = []
        for seq in sequences:
            if self.add_bos_eos:
                seq = [self.token_bos] + seq + [self.token_eos]
            # print('seq',seq)
            unk_index = self.token2idx[self.token_unk]
            result.append([self.token2idx.get(token, unk_index) for token in seq])
            # print('result',result)
        return result

    def numerize_label_sequences(self,
                                 sequences: List[str]) -> List[int]:
        """
        Convert label sequence to label-index sequence
        ``['O', 'O', 'B-ORG'] -> [0, 0, 2]``

        Args:
            sequences: label sequence, list of str

        Returns:
            label-index sequence, list of int
        """
        return [self.label2idx[label] for label in sequences]

    def reverse_numerize_label_sequences(self, sequences, **kwargs):
        if self.multi_label:
            return self.multi_label_binarizer.inverse_transform(sequences)
        else:
            return [self.idx2label[label] for label in sequences]


if __name__ == '__main__':
    DATAPATH = './data/cnews/'
    # test_x, test_y = read_data_file(os.path.join(DATAPATH, 'cnews.test.txt'), 100)
    # train_x, train_y = read_data_file(os.path.join(DATAPATH, 'cnews.train.txt'), 100)
    val_x, val_y = read_data_file(os.path.join(DATAPATH, 'cnews.val.txt'), 100)

    p = ClassificationProcessor(sequence_length=100,
                                add_bos_eos=False)

    p.analyze_corpus(val_x, val_y)
    x_idx = p.process_x_dataset(val_x)
    y_idx = p.process_y_dataset(val_y)

    # print(val_x[0])
    print(x_idx)
    print(type(x_idx))
    # print(val_y[0])
    print(y_idx[0])
    print(x_idx.shape)
