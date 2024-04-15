
import numpy as np
import random
import os, errno
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


# RNN based language model
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)


class Multiheadattention(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int):
        super().__init__()

        # embedding_dim, d_model, 512 in paper
        self.hidden_dim = hidden_dim
        # 8 in paper
        self.num_head = num_head
        # head_dim, d_key, d_query, d_value, 64 in paper (= 512 / 8)
        self.head_dim = hidden_dim // num_head
        self.scale = torch.sqrt(torch.FloatTensor()).to(device)

        self.fcQ = nn.Linear(hidden_dim, hidden_dim)
        self.fcK = nn.Linear(hidden_dim, hidden_dim)
        self.fcV = nn.Linear(hidden_dim, hidden_dim)
        self.fcOut = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, srcQ, srcK, srcV, mask=None):
        ##### SCALED DOT PRODUCT ATTENTION ######

        # input : (bs, seq_len, hidden_dim)
        Q = self.fcQ(srcQ)
        K = self.fcK(srcK)
        V = self.fcV(srcV)

        Q = rearrange(
            Q, 'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim', num_head=self.num_head)
        K_T = rearrange(
            K, 'bs seq_len (num_head head_dim) -> bs num_head head_dim seq_len', num_head=self.num_head)
        V = rearrange(
            V, 'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim', num_head=self.num_head)

        attention_energy = torch.matmul(Q, K_T)
        # attention_energy : (bs, num_head, q_len, k_len)

        if mask is not None:
            '''
            mask.shape
            if padding : (bs, 1, 1, k_len)
            if lookahead : (bs, 1, q_len, k_len)
            '''
            attention_energy = torch.masked_fill(attention_energy, (mask == 0), -1e+4)

        attention_energy = torch.softmax(attention_energy, dim=-1)

        result = torch.matmul(self.dropout(attention_energy), V)
        # result (bs, num_head, seq_len, head_dim)

        ##### END OF SCALED DOT PRODUCT ATTENTION ######

        # CONCAT
        result = rearrange(result, 'bs num_head seq_len head_dim -> bs seq_len (num_head head_dim)')
        # result : (bs, seq_len, hidden_dim)

        # LINEAR

        result = self.fcOut(result)

        return result


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, inner_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.multiheadattention1 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm1 = nn.LayerNorm(hidden_dim)
        self.multiheadattention2 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, inner_dim)
        self.layerNorm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, input, enc_output, paddingMask, lookaheadMask):
        # input : (bs, seq_len, hidden_dim)
        # enc_output : (bs, seq_len, hidden_dim)

        # first multiheadattention
        output = self.multiheadattention1(input, input, input, lookaheadMask)
        output = self.dropout1(output)
        output = output + input
        output = self.layerNorm1(output)

        # second multiheadattention
        output_ = self.multiheadattention2(output, enc_output, enc_output, paddingMask)
        output_ = self.dropout2(output_)
        output = output_ + output
        output = self.layerNorm2(output)

        # Feedforward Network
        output_ = self.ffn(output)
        output_ = self.dropout3(output_)
        output = output + output_
        output = self.layerNorm3(output)

        return output


class Decoder(nn.Module):
    def __init__(self, N, hidden_dim, num_head, inner_dim, max_length=100):
        super().__init__()

        # N : number of encoder layer repeated
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.dec_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)])

        self.dropout = nn.Dropout(p=0.1)

        self.finalFc = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, input, enc_src, enc_output):
        # input = dec_src : (bs, seq_len)
        # enc_src : (bs, seq_len)
        # enc_output : (bs, seq_len,hidden_dim)

        lookaheadMask = makeMask(input, option='lookahead')
        paddingMask = makeMask(enc_src, option='padding')

        # embedding layer
        output = self.embedding(input)
        # output = (bs, seq_len, hidden_dim)

        # Positional Embedding
        # output = pos_embed(output)

        # Dropout
        output = self.dropout(output)

        # N decoder layer
        for layer in self.dec_layers:
            output = layer(output, enc_output, paddingMask, lookaheadMask)
        # output : (bs, seq_len, hidden_dim)

        logits = self.finalFc(output)
        # logits : (bs, seq_len, VOCAB_SIZE)
        output = torch.softmax(logits, dim=-1)

        output = torch.argmax(output, dim=-1)
        # output : (bs, seq_len), dtype=int64

        return logits, output



