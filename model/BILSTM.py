# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

class BILSTM_Model(nn.Module):

    model_name = "BILSTM_Model"
    """
    model arc
    """
    def __init__(self, seq_len, embed_len, token2idx_len):
        super(BILSTM_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.max_pool_layer = nn.MaxPool1d(2)

        # lstm units
        self.bidirectional = True
        self.lstm_params_init = False
        self.rnn_layer_num = 1
        self.hidden_size = 10
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn_layer = nn.LSTM(embed_len, self.hidden_size, self.rnn_layer_num,
                                  batch_first=True,
                                  bidirectional=self.bidirectional)
        self.dense_layer = nn.Linear(self.num_directions*self.hidden_size, 10)

    def forward(self, x):
        x = self.embedding_layer(x.long())
        x = x.permute(0, 2, 1)

        # print(x.shape)
        x = self.max_pool_layer(x)
        x = x.permute(0, 2, 1)
        # lstm init
        if self.lstm_params_init:
            h0 = torch.randn(self.num_directions*self.rnn_layer_num, x.shape[0], self.hidden_size)
            c0 = torch.randn(self.num_directions*self.rnn_layer_num, x.shape[0], self.hidden_size)
            x, _ = self.rnn_layer(x, (h0, c0))
        else:
            x, output = self.rnn_layer(x)
        x = x[:, -1, :]
        x = self.dense_layer(x)
        return x
