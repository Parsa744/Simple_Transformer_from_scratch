import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import T


class selfAttention(nn.Module):
    def __init__(self, input_size):
        super(selfAttention, self).__init__()
        self.input_size = input_size
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        cross = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.input_size)
        attention = torch.softmax(cross, dim=-1)
        out = torch.matmul(attention, V)
        return out



class feedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(feedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out

class crossAttention(nn.Module):
    def __init__(self, input_size,context_size):
        super(crossAttention, self).__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(context_size, input_size)
        self.value = nn.Linear(context_size, input_size)
    def forward(self, x, context):
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)
        cross = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.input_size)
        attention = torch.softmax(cross, dim=-1)
        out = torch.matmul(attention, V)
        return out


class transformerEncoderLayer(nn.Module):
    def __init__(self, input_size, ff_hidden_size):
        super(transformerEncoderLayer, self).__init__()
        self.self_attention = selfAttention(input_size)
        self.feed_forward = feedForwardNN(input_size, ff_hidden_size, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x):
        attn_out = self.self_attention(x)
        ff_out = self.feed_forward(attn_out)
        # is this right?
        return ff_out


class transformerDecoderLayer(nn.Module):
    def __init__(self, input_size, ff_hidden_size,context_size=None):
        super(transformerDecoderLayer, self).__init__()
        self.self_attention = selfAttention(input_size)
        self.cross_attention = crossAttention(input_size,context_size)
        self.feed_forward = feedForwardNN(input_size, ff_hidden_size, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.norm3 = nn.LayerNorm(input_size)

    def forward(self, x, context):
        attn_out = self.self_attention(x)
        cross_attn_out = self.cross_attention(attn_out, context)
        ff_out = self.feed_forward(cross_attn_out)
        return ff_out


class transformer(nn.Module):
    def __init__(self, input_size, ff_hidden_size, num_encoder_layers, num_decoder_layers):
        super(transformer, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [transformerEncoderLayer(input_size, ff_hidden_size) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [transformerDecoderLayer(input_size, ff_hidden_size,context_size) for _ in range(num_decoder_layers)]
        )

    def forward(self, src, tgt):
        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)

        return dec_out

def test():
    model = transformer(input_size=512, ff_hidden_size=2048, num_encoder_layers=6, num_decoder_layers=6)
    src = torch.rand((10, 32, 512))  # (sequence_length, batch_size, input_size)
    tgt = torch.rand((20, 32, 512))  # (sequence_length, batch_size, input_size)
    out = model(src, tgt)
    print(out.shape)  # Expected output shape: (20, 32, 512)
    #assert out.shape == (20, 32, 512)

def testCrossAttention():
    model = crossAttention(input_size=512,context_size=512)
    x = torch.rand((10, 32, 512))  # (sequence_length, batch_size, input_size)
    context = torch.rand((10, 64, 512))  # (sequence_length, batch_size, context_size)
    out = model(x, context)
    print(out.shape)  # Expected output shape: (10, 32, 512)
    assert out.shape == (10, 32, 512)
# test()
def main():
    testCrossAttention()
if __name__ == "__main__":
    main()
