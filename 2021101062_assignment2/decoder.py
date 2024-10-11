import torch
import torch.nn as nn

from encoder import MultiHeadAttention, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=1024, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_masked_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, encoder_output, src_mask=None, tgt_mask=None):

        out1 = self.self_masked_attention(X, X, X, tgt_mask)
        X = X + self.dropout(out1)
        X = self.norm_1(X)

        out2 = self.encoder_decoder_attention(X, encoder_output, encoder_output, src_mask)
        X = X + self.dropout(out2)
        X = self.norm_2(X)

        out3 = self.feed_forward(X)
        X = X + self.dropout(out3)
        X = self.norm_3(X)

        return X
    
def generate_mask(src, tgt):
    
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    subseq_mask = 1-torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1)
    tgt_mask = torch.logical_and(tgt_mask, subseq_mask)

    return src_mask, tgt_mask
