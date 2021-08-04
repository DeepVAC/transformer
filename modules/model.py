import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, ntoken: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(ntoken, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TransformerEncoderNet(nn.Module):
    def __init__(self, ntoken, emb_size, nhead, dim_feedforward, nlayers, dropout=0.1):
        super(TransformerEncoderNet, self).__init__()
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.src_tok_emb = TokenEmbedding(ntoken, emb_size)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.emb_size = emb_size
        self.decoder = nn.Linear(emb_size, ntoken)
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        output = self.transformer_encoder(src_emb, src_mask)
        output = self.decoder(output)
        return output

class TransformerNet(nn.Module):
    def __init__(self,num_encoder_layers: int,num_decoder_layers: int,emb_size: int,nhead: int,src_vocab_size: int,tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(TransformerNet, self).__init__()
        self.transformer = Transformer(d_model=emb_size,nhead=nhead,num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.init_weights()

    def forward(self,src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor, tgt_padding_mask: Tensor,memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        x = self.generator(outs)
        return x

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
