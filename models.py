import torch
import torch.nn as nn
import numpy as np
import math


class SelfAttention(nn.Module):
    def __init__(self, input_size, numer_heads=1):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(input_size, input_size)
        self.K = nn.Linear(input_size, input_size)
        self.V = nn.Linear(input_size, input_size)

    def forward(self, inputs):
        Q = self.Q(inputs)
        K = self.K(inputs)
        V = self.V(inputs)
        # print("Q,K,V",Q,K,V)
        A = torch.softmax(torch.matmul(Q, torch.transpose(K, -2, -1)) / math.sqrt(K.size(-1)), dim=-1)
        Attention = torch.matmul(A, V)
        return Attention


class CrossAttention(nn.Module):
    def __init__(self, input_size, contex_size, hidden_dim):
        super(CrossAttention, self).__init__()
        self.Q = nn.Linear(input_size, hidden_dim)
        self.K = nn.Linear(contex_size, hidden_dim)
        self.V = nn.Linear(contex_size, hidden_dim)

    def forward(self, inputs, context):
        Q = self.Q(inputs)
        K = self.K(context)
        V = self.V(context)
        # print("Q,K,V shapes",Q.shape,K.shape,V.shape)
        A = torch.softmax(torch.matmul(Q, torch.transpose(K, -2, -1)) / math.sqrt(K.size(-1)), dim=-1)
        Attention = torch.matmul(A, V)
        return Attention


class FeedForwaed(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(FeedForwaed, self).__init__()
        self.f1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(hidden_dim, input_size)

    def forward(self, inputs):
        f1 = self.f1(inputs)
        relu = self.relu(f1)
        f2 = self.f2(relu)
        return f2


class SimpleTransformer(nn.Module):
    def __init__(self, input_size, contex_size, hidden_dim):
        super(SimpleTransformer, self).__init__()

        self.PositionalEncoding = PositionalEncoding(300, input_size)
        self.SelfAtt1 = SelfAttention(input_size)
        self.FF1 = FeedForwaed(input_size, hidden_dim)
        self.SelfAtt2 = SelfAttention(input_size)
        self.CrossAtt1 = CrossAttention(input_size, contex_size, hidden_dim)
        self.FF2 = FeedForwaed(hidden_dim, input_size)
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.ln3 = nn.LayerNorm(input_size)
        self.ln4 = nn.LayerNorm(input_size)
        self.ln5 = nn.LayerNorm(input_size)

    def forward(self, inputs, context, norm=True):
        # here will be positional encoding
        PE = self.PositionalEncoding[:inputs.size(1), :].to(inputs.device)
        inputs = inputs + PE.unsqueeze(0)
        Att1 = inputs + self.SelfAtt1(inputs)  # ADD original input back
        # normalize Att1
        Att1 = self.ln1(Att1)
        FF1 = Att1 + self.FF1(Att1)  # ADD Att1 back
        if norm:
            FF1 = self.ln2(FF1)
        Att2 = FF1 + self.SelfAtt2(FF1)  # ADD FF1 back
        if norm:
            Att2 = self.ln3(Att2)
        CrossAtt = Att2 + self.CrossAtt1(Att2, context)  # ADD Att2 back
        if norm:
            CrossAtt = self.ln4(CrossAtt)
        FF2 = CrossAtt + self.FF2(CrossAtt)
        if norm:
            FF2 = self.ln5(FF2)
        return FF2


def PositionalEncoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return PE
