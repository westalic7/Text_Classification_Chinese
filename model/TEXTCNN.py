# -*- coding:utf-8 -*-


import torch
import torch.nn.functional as F
from dropout.Spatial_Dropout import Spatial_Dropout
from torch import nn


class TEXTCNN_Model(nn.Module):
    model_name = "TEXTCNN_Model"
    """
    model arc
        state: perform bad
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(TEXTCNN_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.spacial_dropout = Spatial_Dropout(drop_prob=0.25)

        self.kernel_sile_list = [2, 3, 4, 5]
        self.conv1d_all = nn.ModuleList(
            [nn.Conv2d(1, 128, kernel_size=(ks, embed_len)) for ks in self.kernel_sile_list])

        self.dropout = nn.Dropout(0.5)
        self.dense_layer = nn.Linear(128 * 4, 10)

    def forward(self, x):
        x = self.embedding_layer(x.long())  # embedding: sequence-dim=1, input:[batch_size, sequence_len, embedding_len]
        x = x.permute(0, 2, 1)
        x = self.spacial_dropout(x)  # spacial_dropout: sequence-dim=2
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1d_all]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.dense_layer(x)

        return x
