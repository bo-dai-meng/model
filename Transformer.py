import math
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torch.utils.data import DataLoader

# 多头自注意力机制中，头数目为2。
def scaled_dot_product(q, k, v, mask = None):
    dim_k = q.size()[-1]
    qk_dot_product = torch.matmul(q, k.transpose(-2, -1))
    dot_product = qk_dot_product / math.sqrt(dim_k)
    if mask is not None:
        dot_product = dot_product.masked_fill(mask==0, -9e15)
    attention = nn.functional.softmax(dot_product, dim=-1)
    values = torch.matmul(attention, v)
    return values

class Attention(nn.Module):
    def __init__(self, dim_input: int, dim_q: int, dim_k: int):
        super().__init__()
        self.linear_q = nn.Linear(dim_input, dim_q)
        self.linear_k = nn.Linear(dim_input, dim_k)
        self.linear_v = nn.Linear(dim_input, dim_k)
    def forward(self, query, key, value, mask=None):
        return scaled_dot_product(self.linear_q(query), self.linear_k(key), self.linear_v(value), mask)

class MutiHeadAtt(nn.Module):
    def __init__(self, num_heads, dim_input, dim_q, dim_k):
        super().__init__()
        self.heads1 = Attention(dim_input, dim_q, dim_k)
        self.heads2 = Attention(dim_input, dim_q, dim_k)
        self.linear = nn.Linear(num_heads * dim_k, dim_input)

    def forward(self, query, key, value, mask=None):
        head1 = self.heads1(query, key, value, mask)
        head2 = self.heads2(query, key, value, mask)
        output = torch.cat([head1, head2], dim=-1)
        return self.linear(output)

def feed_forward(input_dim, intermediate_dim):
    return nn.Sequential(
        nn.Linear(input_dim, intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, input_dim)
    )

class AddNorm(nn.Module):
    def __init__(self, sublayer, dim, dropout_rate=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, tensors, *args, **kwargs):
        return self.norm(tensors + self.dropout(self.sublayer(tensors, *args, **kwargs)))

class PositionalEncoding(nn.Module):
    # 位置编码，在输入序列中加入位置信息
    def __init__(self, num_hiddens, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        query_dim = key_dim = int(max(input_dim / num_heads, 1))
        self.multi_head_attention = AddNorm(
            MutiHeadAtt(num_heads, input_dim, query_dim, key_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )
        self.feedforward_network = AddNorm(
            feed_forward(input_dim, feedforward_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, src, mask=None):
        output = self.multi_head_attention(src, src, src, mask)
        final_output = self.feedforward_network(output)
        return final_output

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        self.layer= nn.ModuleList(
            [TransformerEncoderLayer(input_dim, num_heads, feedforward_dim, dropout_rate) for i in range(num_layers)]
        )
        self.pos_lay = PositionalEncoding(input_dim)
    def forward(self, src, mask=None):
        seq_len, input_dim = src.size(1), src.size(2)
        input = (src + self.pos_lay(src))
        for layer in self.layer:
            enc_output = layer(input, mask)
        return enc_output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        query_dim = key_dim = int(max(input_dim / num_heads, 1))
        self.self_attention = AddNorm(
            MutiHeadAtt(num_heads, input_dim, query_dim, key_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )
        self.cross_attention = AddNorm(
            MutiHeadAtt(num_heads, input_dim, query_dim, key_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )
        self.feed_forward = AddNorm(
            feed_forward(input_dim, feedforward_dim),
            dim=input_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, target, enc_output, mask):
        output = self.self_attention(target, target, target, mask)
        output = self.cross_attention(output, enc_output, enc_output)
        return self.feed_forward(output)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layer, input_dim, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(input_dim, num_heads, feedforward_dim, dropout_rate) for i in range(num_layer)]
        )
        self.FC = nn.Linear(input_dim, 1)
        self.Pos = PositionalEncoding(input_dim)
    def forward(self, target, enc_output, mask):
        input = self.Pos(target) + target
        for layer in self.layers:
            FC_input = layer(input, enc_output, mask)
        return self.FC(FC_input)

class Transformer_Encoder_Decoder(nn.Module):
    def __init__(self, num_layer: int, input_dim: int, num_heads: int, feedforward_dim: int, dropout_rate):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_layer,
            input_dim=input_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout_rate=dropout_rate
        )
        self.decoder = TransformerDecoder(
            num_layer=num_layer,
            input_dim=input_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout_rate=dropout_rate
        )
    def forward(self, src_input, target, mask):
        enc_output = self.encoder(src_input)
        return self.decoder(target, enc_output, mask)[:, -1, :]

# sequence mask:并行训练时，保证自注意力层的当前时间步的输出只能看到其之前的输入。
def get_masks(timestep):
    return torch.tril(torch.ones(timestep, timestep), diagonal=0).bool()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)

