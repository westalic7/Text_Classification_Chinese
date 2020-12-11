# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Bag of Tricks for Efficient Text Classification'''


class FastText_Model(nn.Module):
    model_name = "FastText_Model"
    """
    model arc
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(FastText_Model, self).__init__()
        self.n_gram_vocab = token2idx_len  # 25049
        self.embed = embed_len
        self.hidden_size = 256
        self.num_classes = 10
        self.dropout = 0.5
        self.embedding_pretrained = None
        self.n_vocab = 0

        if self.embedding_pretrained is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            # self.embedding = nn.Embedding(self.n_vocab, self.embed, padding_idx=self.n_vocab - 1)
            self.embedding_layer = nn.Embedding(self.n_gram_vocab, self.embed)

        self.embedding_ngram2 = nn.Embedding(self.n_gram_vocab, self.embed)
        self.embedding_ngram3 = nn.Embedding(self.n_gram_vocab, self.embed)
        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.embed * 3, self.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):

        out_word = self.embedding_layer(x.long())
        # print('out_word',out_word.shape)
        out_bigram = self.embedding_ngram2(x.long())
        # print(out_bigram.shape)
        out_trigram = self.embedding_ngram3(x.long())
        # print(out_trigram.shape)
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        # print(out.shape)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
