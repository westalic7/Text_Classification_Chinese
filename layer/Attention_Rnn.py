# -*- coding:utf-8 -*-
# @Time    : 2020/7/7 4:08 PM
# @Author  : Wang Xinle


import torch
from torch import nn
import torch.nn.functional as F

class Attention_Rnn(nn.Module):

    model_name = "Attention_Rnn"
    """
    model arc
    """
    def __init__(self, sequence_dim=1, return_attention=False, squeeze_out=False):
        super(Attention_Rnn, self).__init__()
        self.sequence_dim = sequence_dim
        self.return_attention = return_attention
        self.squeeze_out = squeeze_out

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, final_state.shape[0]*final_state.shape[2], 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        if self.squeeze_out:
            context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        else:
            context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2))
        return context, soft_attn_weights.data.numpy()

    def forward(self, x, fh=None):

        assert len(x.shape)==3
        if self.sequence_dim==1:
            pass
        elif self.sequence_dim==2:
            x = x.permute(0, 2, 1)
        else:
            print('sequence dim should be dim-1')

        if fh is None:
            fh = nn.Parameter(torch.randn(1, x.shape[0], x.shape[2]),requires_grad=True)

        attn_x, attention= self.attention_net(x,fh)
        if self.return_attention:
            return attn_x, attention
        else:
            return attn_x