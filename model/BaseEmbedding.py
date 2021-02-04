# -*- coding:utf-8 -*-

import torch.nn.functional as F
from torch import nn


class BaseEmbedding_Model(nn.Module):
    model_name = "Base_Embedding"
    """
    model arc
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(BaseEmbedding_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.max_pool_layer = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(embed_len, 10)

    def forward(self, x):
        x = self.embedding_layer(x.long())
        x = x.permute(0, 2, 1)
        x = self.max_pool_layer(x).squeeze()
        x = F.relu(x)
        x = self.fc(x)
        return x
