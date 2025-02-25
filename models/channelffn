import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAwareFFN(nn.Module):
    """
    Channel-Aware Feed Forward Network.
    
    This FFN variant applies channel-specific weights to process different 
    feature channels adaptively.
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation=F.relu, 
                 channel_attention=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.channel_attention = channel_attention
        
        # Standard FFN layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        # Channel attention mechanism
        if channel_attention:
            self.channel_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 8),
                nn.ReLU(),
                nn.Linear(d_model // 8, d_model),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Apply channel attention if enabled
        if self.channel_attention:
            # Global average pooling along sequence dimension
            channel_context = x.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]
            channel_weights = self.channel_gate(channel_context)  # [batch_size, 1, d_model]
            x = x * channel_weights  # Channel-wise modulation
        
        # Standard FFN forward
        output = self.linear1(x)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.linear2(output)
        
        return output
