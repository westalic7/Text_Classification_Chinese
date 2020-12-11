# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn


class CNN_GRU_Model(nn.Module):
    model_name = "CNN_GRU"
    """
    model arc
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(CNN_GRU_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)

        self.channel = 32
        self.conv1d_layer = nn.Conv1d(in_channels=embed_len,
                                      out_channels=self.channel,
                                      kernel_size=3,
                                      padding=1)

        self.max_pool_layer = nn.MaxPool1d(2)

        # lstm units
        self.bidirectional = False
        self.lstm_params_init = False
        self.rnn_layer_num = 1
        self.hidden_size = 10
        self.num_directions = 2 if self.bidirectional else 1
        self.lstm_layer = nn.GRU(self.channel, self.hidden_size, self.rnn_layer_num,
                                 batch_first=True,
                                 bidirectional=self.bidirectional)
        self.dense_layer = nn.Linear(self.num_directions * self.hidden_size, 10)

    def forward(self, x):
        x = self.embedding_layer(x.long())
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = F.relu(self.conv1d_layer(x))
        # print(x.shape)
        x = self.max_pool_layer(x)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        # lstm init
        # print(x.shape)
        if self.lstm_params_init:
            h0 = torch.randn(self.num_directions * self.rnn_layer_num, x.shape[0], self.hidden_size)
            c0 = torch.randn(self.num_directions * self.rnn_layer_num, x.shape[0], self.hidden_size)
            x, _ = self.lstm_layer(x, (h0, c0))
        else:
            x, _ = self.lstm_layer(x)
        x = x[:, -1, :]
        x = self.dense_layer(x)
        return x
