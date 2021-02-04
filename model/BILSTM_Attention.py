# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class BILSTM_Attention_Model(nn.Module):
    model_name = "BILSTM_Attention_Model"
    """
    model arc
    """

    def __init__(self, seq_len, embed_len, token2idx_len):
        super(BILSTM_Attention_Model, self).__init__()

        self.embedding_layer = nn.Embedding(token2idx_len, embed_len)
        self.n_hidden = 108
        self.num_classes = 10
        self.lstm = nn.LSTM(embed_len, self.n_hidden, bidirectional=True)
        self.out = nn.Linear(self.n_hidden * 2, self.num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, x):
        input = self.embedding_layer(x.long())  # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        hidden_state = torch.zeros(1 * 2, len(x), self.n_hidden,
                                   requires_grad=True)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1 * 2, len(x), self.n_hidden,
                                 requires_grad=True)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]

        return self.out(attn_output)
