# -*- coding:utf-8 -*-


import torch
import torch.nn.functional as F
from dropout.Spatial_Dropout import Spatial_Dropout
from layer.Attention_Rnn import Attention_Rnn
from torch import nn


class R_CNN_Model(nn.Module):
    model_name = "R_CNN_Model"
    """
    model arc
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(R_CNN_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.max_pool_layer = nn.MaxPool1d(2)

        self.spacial_dropout = Spatial_Dropout(drop_prob=0.5)

        self.bidirectional = True
        self.rnn_params_init = False
        self.rnn_layer_num = 1
        self.hidden_size = 64
        self.num_directions = 2 if self.bidirectional else 1
        self.BiGRU = nn.GRU(embed_len, self.hidden_size, self.rnn_layer_num,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        self.conv1d = nn.Conv1d(self.num_directions * self.rnn_layer_num * self.hidden_size,
                                128,
                                kernel_size=2)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.attn_rnn = Attention_Rnn()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dense_layer_10 = nn.Linear(384, 120)
        self.dense_layer_11 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.embedding_layer(x.long())  # embedding: sequence-dim=1, input:[batch_size, sequence_len, embedding_len]
        x = x.permute(0, 2, 1)

        # main branch
        x = self.spacial_dropout(x)  # spacial_dropout: sequence-dim=2
        x = x.permute(0, 2, 1)
        if self.rnn_params_init:
            h0 = torch.randn(self.num_directions * self.rnn_layer_num, x.shape[0], self.hidden_size)
            x, (hn) = self.BiGRU(x, h0)  # spacial_dropout: sequence-dim=1
        else:
            x, (hn) = self.BiGRU(x)  # spacial_dropout: sequence-dim=1
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d(x))  # conv1d: sequence-dim=2

        # sub branch
        # 'global max pool'
        x1 = self.max_pool(x)
        x1 = x1.flatten(start_dim=1)

        # 'attention weighted layer'
        x2 = x.permute(0, 2, 1)
        # print(x2.shape)
        # print(hn.shape)
        x2 = self.attn_rnn(x2, hn)
        # print(x2.shape)
        x2 = x2.flatten(start_dim=1)

        # 'global avg pool'
        x3 = self.avg_pool(x)
        x3 = x3.flatten(start_dim=1)

        # main branch
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.dense_layer_10(x))
        x = self.dense_layer_11(x)

        return x
