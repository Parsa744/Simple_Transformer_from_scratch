import torch
import torch.nn as nn
import numpy as np
import math

class SelfAttention(nn.Module):
    def __init__(self, input_size,numer_heads=1):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(input_size, input_size)
        self.K = nn.Linear(input_size, input_size)
        self.V = nn.Linear(input_size, input_size)
    def forward(self,inputs):
        Q = self.Q(inputs)
        K = self.K(inputs)
        V = self.V(inputs)
        #print("Q,K,V",Q,K,V)
        A = torch.softmax(torch.matmul(Q, torch.transpose(K, -2, -1)) / math.sqrt(K.size(-1)), dim=-1)
        Attention = torch.matmul(A,V)
        return Attention

class CrossAttention(nn.Module):
    def __init__(self, input_size,contex_size,hidden_dim):
        super(CrossAttention, self).__init__()
        self.Q = nn.Linear(input_size,hidden_dim )
        self.K = nn.Linear(contex_size,hidden_dim )
        self.V = nn.Linear(contex_size,hidden_dim )
    def forward(self,inputs,context):
        Q = self.Q(inputs)
        K = self.K(context)
        V = self.V(context)
        #print("Q,K,V shapes",Q.shape,K.shape,V.shape)
        A = torch.softmax(torch.matmul(Q, torch.transpose(K, -2, -1)) / math.sqrt(K.size(-1)), dim=-1)
        Attention = torch.matmul(A,V)
        return Attention

class FeedForwaed(nn.Module):
    def __init__(self, input_size,hidden_dim):
        super(FeedForwaed, self).__init__()
        self.f1 = nn.Linear(input_size,hidden_dim)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(hidden_dim,input_size)


    def forward(self,inputs):
        f1 = self.f1(inputs)
        relu = self.relu(f1)
        f2 = self.f2(relu)
        return f2



class SimpleTransformer(nn.Module):
    def __init__(self, input_size,contex_size,hidden_dim):
        super(SimpleTransformer, self).__init__()
        
        self.SelfAtt1 = SelfAttention(input_size)
        self.FF1 = FeedForwaed(input_size,hidden_dim)
        self.SelfAtt2 = SelfAttention(input_size)
        self.CrossAtt1 = CrossAttention(input_size,contex_size,hidden_dim)
        self.FF2 = FeedForwaed(hidden_dim,input_size)
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.ln3 = nn.LayerNorm(input_size)
        self.ln4 = nn.LayerNorm(input_size)
        self.ln5 = nn.LayerNorm(input_size)

    def forward(self,inputs,context,norm=True):
        Att1 = inputs + self.SelfAtt1(inputs)  # ADD original input back
        # normalize Att1
        Att1 = self.ln1(Att1)
        FF1 = Att1 + self.FF1(Att1)  # ADD Att1 back
        if norm:
            FF1 = self.ln2(FF1)
        Att2 = FF1 + self.SelfAtt2(FF1)  # ADD FF1 back
        if norm:
            Att2 = self.ln3(Att2)
        CrossAtt = Att2 + self.CrossAtt1(Att2,context)  # ADD Att2 back
        if norm:
            CrossAtt = self.ln4(CrossAtt)
        FF2 = CrossAtt + self.FF2(CrossAtt)
        if norm:
            FF2 = self.ln5(FF2)
        return FF2

def testCrossAttention():

    contex_size = 5
    input_size = 3
    hidden_dim = 10
    batch_size = 1
    seq_len_q = 10
    seq_len_k = 10
    input_matrix = torch.rand(batch_size, seq_len_q, input_size)

    #print(input_matrix)
    AttentionHead = CrossAttention(input_size,contex_size,hidden_dim)
    context = torch.rand(batch_size,seq_len_k,contex_size)
    #print('shape of input_matrix:',input_matrix.shape)
    #print('shape of context:',context.shape)
    #print(AttentionHead.forward(input_matrix,context))
    #print('shape of output:',AttentionHead.forward(input_matrix,context).shape)

def testFF():

    input_size = 3
    
    hidden_dim = 20

    seq_len_q = 10

    input_matrix = torch.rand(seq_len_q, input_size)
    ff = FeedForwaed(input_size,hidden_dim)
    x = ff.forward(input_matrix)
    #print(x)
    #print(np.shape(x))
    #print(np.shape(input_matrix))
    
def testSimpleTransformer():

    contex_size = 5
    input_size = 3
    hidden_dim = 20
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 10
    input_matrix = torch.rand(batch_size, seq_len_q, input_size)

    #print(input_matrix)
    transformer = SimpleTransformer(input_size,contex_size,hidden_dim)
    context = torch.rand(batch_size,seq_len_k,contex_size)
    print(transformer.forward(input_matrix,context))

if __name__ == "__main__":
    testSimpleTransformer()


        
