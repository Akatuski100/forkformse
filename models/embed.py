import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

###############################################################################
# 1. Channel Positional Encoding
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAwareFFLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        """
        A simple two-layer feed forward network.
        Args:
            input_dim: The size of the concatenated input (token conv output + channel encoding)
            hidden_dim: The size of the hidden layer.
            output_dim: The desired output dimension (should match d_model).
            dropout: Dropout rate applied after the activation.
        """
        super(ChannelAwareFFLayer, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.ff(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=1, hidden_dim=None):
        """
        TokenEmbedding with integrated channel-aware feed forward layer.
        Args:
            c_in: Number of input channels.
            d_model: The model dimension.
            m: Parameter for ChannelPositionalEmbedding (generates m+1 values per channel).
            hidden_dim: If provided, use ChannelAwareFFLayer with this hidden dimension;
                        otherwise, fall back to a single linear projection.
        """
        super(TokenEmbedding, self).__init__()
        self.m = int(m)  # Ensure m is an integer
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular'
        )
        # The concatenated dimension: token conv output (d_model) + channel encoding (c_in*(m+1))
        input_dim = d_model + c_in * (self.m + 1)
        if hidden_dim is None:
            # Default to a simple linear projection.
            self.channel_ffn = nn.Linear(input_dim, d_model)
        else:
            # Use the more expressive channel-aware feed forward network.
            self.channel_ffn = ChannelAwareFFLayer(input_dim, hidden_dim, d_model)
        
        # Initialize convolution weights.
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, channel_encoding=None):
        # x shape: (batch, seq_len, c_in)
        # Apply convolution: permute to (batch, c_in, seq_len) and then back to (batch, seq_len, d_model)
        x_emb = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        
        if channel_encoding is not None:
            # Ensure channel_encoding has shape (batch, seq_len, c_in*(m+1))
            if channel_encoding.dim() == 2:
                channel_encoding = channel_encoding.unsqueeze(0).expand(x_emb.size(0), -1, -1)
            # Concatenate along the last dimension.
            x_emb = torch.cat([x_emb, channel_encoding], dim=-1)
            # Pass through the channel-aware feed forward network.
            x_emb = self.channel_ffn(x_emb)
        return x_emb


class ChannelPositionalEmbedding(nn.Module):
    """
    Generates a sinusoidal channel positional encoding.
    It creates a (c_in, ma+1) matrix (with ma=8), flattens it to (1, c_in*(ma+1)),
    and repeats it n times to yield a (n, c_in*(ma+1)) tensor.
    """
    def __init__(self, c_in, m=1):
        super(ChannelPositionalEmbedding, self).__init__()
        self.c_in = c_in
        self.m = int(m)  #produces (m+1) columns per channel.
        pe = torch.zeros(c_in, self.m + 1).float()
        pe.requires_grad = False  # fixed encoding
        position = torch.arange(0, c_in).float().unsqueeze(1)  # shape: (c_in, 1)
        div_term = torch.exp(torch.arange(0, self.m + 1, 2).float() * -(math.log(10000.0) / (self.m + 1)))
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].size(1)])
        if (self.m + 1) > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe)

    def forward(self, n):
        # Flatten to (1, c_in*(ma+1)) then repeat n times.
        flat = self.pe.flatten().unsqueeze(0)
        return flat.repeat(n, 1)


###############################################################################
# 3. Modified DataEmbedding
###############################################################################

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x is used only to determine the sequence length.
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13
        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        return hour_x + weekday_x + day_x + month_x + minute_x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)
    def forward(self, x):
        return self.emb(x).detach()

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=1, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        print(f"DEBUG: Received m = {m} (type: {type(m)}) in DataEmbedding", flush=True)
        self.m = int(m)  # Ensure m is an integer
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, m=self.m, hidden_dim=d_model * 2)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != 'timeF'
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        batch_size, seq_len, c_in = x.shape
        
        # Assume you have the desired m available, e.g., self.m or an argument passed to DataEmbedding
        channel_encoder = ChannelPositionalEmbedding(c_in, m=self.m).to(x.device)
        channel_encoding = channel_encoder(seq_len)

        # Compute the Channel Encoding (shape: (seq_len, c_in*(ma+1)))
        channel_encoding = channel_encoder(seq_len)
        
        # Pass the channel encoding to the token embedding.
        x = (
            self.value_embedding(x, channel_encoding=channel_encoding)
            + self.position_embedding(x)
            + self.temporal_embedding(x_mark)
        )
        return self.dropout(x)

