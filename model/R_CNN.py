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

        self.bidirectional = True
        self.rnn_layer_num = 1
        self.hidden_size = 128
        self.num_directions = 2 if self.bidirectional else 1
        self.BiGRU = nn.GRU(embed_len, self.hidden_size, self.rnn_layer_num,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            )

        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(self.hidden_size * self.num_directions + 100, 10)

    def forward(self, x):
        embed = self.embedding_layer(
            x.long())  # embedding: sequence-dim=1, input:[batch_size, sequence_len, embedding_len]

        x, (hn) = self.BiGRU(embed)  # spacial_dropout: sequence-dim=1
        x = torch.cat((embed, x), 2)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x).squeeze()
        x = self.fc(x)

        return x
