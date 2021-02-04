# -*- coding:utf-8 -*-


from torch import nn


class BIGRU_Model(nn.Module):
    model_name = "BIGRU_Model"
    """
    model arc
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(BIGRU_Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token2idx_len, embedding_dim=embed_len)

        # lstm units
        self.bidirectional = True
        self.rnn_layer_num = 1
        self.hidden_size = 128
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn_layer = nn.GRU(embed_len, self.hidden_size, self.rnn_layer_num,
                                batch_first=True,
                                bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.num_directions * self.hidden_size, 10)

    def forward(self, x):
        x = self.embedding_layer(x.long())
        x, output = self.rnn_layer(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
