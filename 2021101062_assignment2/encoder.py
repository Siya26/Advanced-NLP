import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)

        for pos in range(int(max_len)):
            for i in range(0, d_model, 2):
                self.encoding[pos, i] = math.sin(pos / 10000 ** ((2 * i)/d_model))
                if i+1 < d_model:
                    self.encoding[pos, i+1] = math.cos(pos / 10000 ** ((2 * i))/d_model)

        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        print(x.shape, self.encoding[:, :x.size(1)].shape)
        x = x + self.encoding[:, :x.size(1)].to(x.device).detach()
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be an integer multiple of num_heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

    
    def self_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(scores, V)
        return output

    def forward(self, Q, K, V, mask=None):

        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = Q.view(Q.size(0), -1, self.num_heads, self.d_k)
        K = K.view(K.size(0), -1, self.num_heads, self.d_k)
        V = V.view(V.size(0), -1, self.num_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = self.self_attention(Q, K, V, mask)

        concat = scores.transpose(1, 2).contiguous().view(Q.size(0), -1, self.d_model)

        output = self.W_O(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=1024, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out1 = self.attention(x, x, x, mask)
        x = x + self.dropout_1(out1)
        x = self.norm_1(x)

        out2 = self.feed_forward(x)
        x = x + self.dropout_2(out2)
        x = self.norm_2(x)

        return x