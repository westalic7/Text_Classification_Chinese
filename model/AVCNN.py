# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from dropout.Spatial_Dropout import Spatial_Dropout
from layer.Attention_Rnn import Attention_Rnn

class AVCNN_Model(nn.Module):

    model_name = "AVCNN_Model"
    """
    model arc
        state: perform bad
    """
    def __init__(self, seq_len, embed_len, token2idx_len):
        super(AVCNN_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.spacial_dropout = Spatial_Dropout(drop_prob=0.25)

        self.conv1d_all = nn.ModuleList([nn.Conv1d(embed_len,300,kernel_size=i+1) for i in range(4)])

        self.feature_layers_all = nn.ModuleList([nn.AdaptiveMaxPool1d(1),
                                   Attention_Rnn(sequence_dim=2),
                                   nn.AdaptiveAvgPool1d(1)])

        self.dense_layer0 = nn.Linear(3600, 144)
        self.dense_layer1 = nn.Linear(144, 10)

    def forward(self, x):
        # tree node 0
        x = self.embedding_layer(x.long())  # embedding: sequence-dim=1, input:[batch_size, sequence_len, embedding_len]
        x = x.permute(0, 2, 1)
        x = self.spacial_dropout(x)  # spacial_dropout: sequence-dim=2

        # tree node 1: cnn
        tensors_feature_all = []
        for feat in self.feature_layers_all:  # concat all other pool1d tensor next
            tensors_conv_all = []
            for conv in self.conv1d_all:  # concat all conv1d tensor first
                x_conv = F.relu(conv(x))
                x_conv = feat(x_conv)
                x_conv = x_conv.squeeze(2)
                tensors_conv_all.append(x_conv)
            tensor_conv_all = torch.cat(tensors_conv_all, dim=1)
            # print(tensor_conv_all.shape)
            tensors_feature_all.append(tensor_conv_all)
        tensor_feature_all = torch.cat(tensors_feature_all, dim=1)
        # print(tensor_feature_all.shape)

        # tree node 2
        x = F.dropout(tensor_feature_all, p=0.7)
        x = F.relu(self.dense_layer0(x))
        x = self.dense_layer1(x)

        return x
