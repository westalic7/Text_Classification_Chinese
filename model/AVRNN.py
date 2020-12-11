# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from dropout.Spatial_Dropout import Spatial_Dropout
from layer.Attention_Rnn import Attention_Rnn

class AVRNN_Model(nn.Module):

    model_name = "AVRNN_Model"
    """
    model arc
    """
    def __init__(self, seq_len, embed_len, token2idx_len):
        super(AVRNN_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.spacial_dropout = Spatial_Dropout(drop_prob=0.5)

        self.bidirectional = True
        self.rnn_params_init = False
        self.rnn_layer_num = 1
        self.hidden_size = 60
        self.num_directions = 2 if self.bidirectional else 1
        self.Bidirection_rnn_0 = nn.GRU(embed_len,
                                        self.hidden_size,
                                        self.rnn_layer_num,
                                        batch_first=True,
                                        bidirectional=self.bidirectional)
        self.Bidirection_rnn_1 = nn.GRU(self.hidden_size*self.num_directions,
                                        self.hidden_size,
                                        self.rnn_layer_num,
                                        batch_first=True,
                                        bidirectional=self.bidirectional)

        self.conv1d = nn.Conv1d(self.num_directions * self.rnn_layer_num * self.hidden_size,
                                128,
                                kernel_size=2)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.attn_rnn = Attention_Rnn()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dense_layer_0 = nn.Linear(960, 144)
        self.dense_layer_1 = nn.Linear(144, 10)

    def forward(self, x):
        # tree node 0
        x = self.embedding_layer(x.long())  # embedding: sequence-dim=1, input:[batch_size, sequence_len, embedding_len]
        x = x.permute(0, 2, 1)
        x = self.spacial_dropout(x)  # spacial_dropout: sequence-dim=2
        x = x.permute(0, 2, 1)

        # tree node 1: rnn concat layer
        # rnn0
        if self.rnn_params_init:
            h0_10 = torch.randn(self.num_directions * self.rnn_layer_num, x.shape[0], self.hidden_size)
            x10, (hn_10) = self.Bidirection_rnn_0(x, h0_10)  # spacial_dropout: sequence-dim=1
        else:
            x10, (hn_10) = self.Bidirection_rnn_0(x)  # spacial_dropout: sequence-dim=1
        # rnn1
        if self.rnn_params_init:
            h0_11 = torch.randn(self.num_directions * self.rnn_layer_num, x.shape[0], self.hidden_size)
            x11, (hn_11) = self.Bidirection_rnn_1(x10, h0_11)  # spacial_dropout: sequence-dim=1
        else:
            x11, (hn_11) = self.Bidirection_rnn_1(x10)  # spacial_dropout: sequence-dim=1

        x = torch.cat([x10, x11], dim=2)
        hn_cat = torch.cat([hn_10, hn_11], dim=2)

        # tree node 2
        # last hidden state layer
        x21 = x[:, -1, :]

        # global max pool
        x22 = self.max_pool(x.permute(0, 2, 1))
        x22 = x22.flatten(start_dim=1)

        # attention weighted layer
        x23 = self.attn_rnn(x, hn_cat)
        x23 = x23.flatten(start_dim=1)

        # global avg pool
        x24 = self.avg_pool(x.permute(0, 2, 1))
        x24 = x24.flatten(start_dim=1)

        x = torch.cat([x21, x22, x23, x24], dim=1)

        # tree node 3
        x = F.dropout(x, p=0.5)
        x = F.relu(self.dense_layer_0(x))
        x = self.dense_layer_1(x)

        return x
