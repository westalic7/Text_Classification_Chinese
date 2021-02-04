# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


class DPCNN_Model(nn.Module):
    model_name = "DPCNN_Model"

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(DPCNN_Model, self).__init__()
        self.channel_num = 196
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)
        self.conv_region = nn.Conv2d(1, self.channel_num, (3, embed_len), stride=1)
        self.conv = nn.Conv2d(self.channel_num, self.channel_num, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.bn = nn.BatchNorm2d(self.channel_num)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.channel_num, 10)

    def forward(self, x):
        # x = x[0]
        x = self.embedding_layer(x.long())
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        ln = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
        x = ln(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        ln = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
        x = ln(x)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
