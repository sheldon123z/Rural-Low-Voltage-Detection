"""
LSTM AutoEncoder for Time Series Anomaly Detection

A classic baseline model using LSTM-based encoder-decoder architecture
for reconstruction-based anomaly detection.

Architecture:
    Encoder: Multi-layer LSTM that compresses input sequence
    Decoder: Multi-layer LSTM that reconstructs from latent space

Reference:
    Malhotra et al. (2015) "Long Short Term Memory Networks for Anomaly Detection"
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    LSTM AutoEncoder for anomaly detection.
    
    Learns to reconstruct normal time series patterns.
    High reconstruction error indicates potential anomaly.
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Hidden dimension and layers
        self.d_model = getattr(configs, 'd_model', 64)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # Latent space dimension
        self.latent_dim = self.d_model // 2
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            batch_first=True,
            dropout=self.dropout if self.e_layers > 1 else 0,
            bidirectional=False
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Linear(self.d_model, self.latent_dim)
        self.expand = nn.Linear(self.latent_dim, self.d_model)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            batch_first=True,
            dropout=self.dropout if self.e_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.c_out)
        
        # Activation
        self.activation = nn.ReLU()
        
    def encode(self, x):
        """
        Encode input sequence to latent representation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, enc_in)
            
        Returns:
            latent: Latent representation of shape (batch, latent_dim)
            hidden: LSTM hidden states for decoder initialization
        """
        # LSTM encoding
        encoder_outputs, (hidden, cell) = self.encoder(x)
        
        # Use final hidden state as latent representation
        # hidden shape: (num_layers, batch, d_model)
        final_hidden = hidden[-1]  # (batch, d_model)
        
        # Bottleneck compression
        latent = self.activation(self.bottleneck(final_hidden))
        
        return latent, (hidden, cell)
    
    def decode(self, latent, seq_len):
        """
        Decode latent representation to reconstructed sequence.
        
        Args:
            latent: Latent tensor of shape (batch, latent_dim)
            seq_len: Target sequence length
            
        Returns:
            reconstructed: Tensor of shape (batch, seq_len, c_out)
        """
        batch_size = latent.size(0)
        
        # Expand latent to decoder input dimension
        expanded = self.activation(self.expand(latent))  # (batch, d_model)
        
        # Repeat for sequence length
        decoder_input = expanded.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, d_model)
        
        # Decode
        decoder_outputs, _ = self.decoder(decoder_input)
        
        # Project to output dimension
        reconstructed = self.output_projection(decoder_outputs)
        
        return reconstructed
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for anomaly detection.
        
        Args:
            x_enc: Input tensor of shape (batch, seq_len, enc_in)
            Other arguments are unused (for API compatibility)
            
        Returns:
            reconstructed: Tensor of shape (batch, seq_len, c_out)
        """
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        else:
            # Default to anomaly detection behavior
            return self.anomaly_detection(x_enc)
    
    def anomaly_detection(self, x_enc):
        """
        Anomaly detection through reconstruction.
        
        Args:
            x_enc: Input tensor of shape (batch, seq_len, enc_in)
            
        Returns:
            reconstructed: Tensor of shape (batch, seq_len, c_out)
        """
        # Encode
        latent, _ = self.encode(x_enc)
        
        # Decode
        reconstructed = self.decode(latent, self.seq_len)
        
        return reconstructed
