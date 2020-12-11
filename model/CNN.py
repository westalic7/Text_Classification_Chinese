# -*- coding:utf-8 -*-

import torch.nn.functional as F
from torch import nn


class CNN_Model(nn.Module):
    model_name = "CNN_Model"
    """
    model arc
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(CNN_Model, self).__init__()
        self.channel = 128
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.conv1d_layer = nn.Conv1d(in_channels=embed_len, out_channels=self.channel, kernel_size=5)
        self.max_pool_layer = nn.AdaptiveMaxPool1d(1)
        # self.dense_layer_0 = nn.Linear(seq_len-4, 1)
        self.dense_layer_1 = nn.Linear(self.channel, 10)
        # self.dense_layer_2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.embedding_layer(x.long())
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d_layer(x))
        # x = F.relu(self.dense_layer_0(x))
        x = self.max_pool_layer(x)
        x = x.flatten(start_dim=1)
        # x = F.relu(self.dense_layer_1(x))
        x = self.dense_layer_1(x)
        return x
