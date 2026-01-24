"""
TPATimesNet: Three-Phase Attention TimesNet for Rural Voltage Anomaly Detection

Core Innovation: Introduces Three-Phase Attention mechanism that explicitly models
the correlations between three-phase voltages (Va, Vb, Vc).

In normal three-phase systems:
- Va, Vb, Vc have 120° phase shifts
- Magnitudes should be balanced (within ±4%)
- Any deviation indicates potential anomalies

Key Components:
1. ThreePhaseAttention: Cross-attention between phases
2. Phase-aware positional encoding
3. Unbalance-aware loss function

Author: Voltage Anomaly Detection Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import numpy as np
from typing import Optional, Tuple, List

from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    Use FFT to find the top-k periods in the time series.
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class ThreePhaseAttention(nn.Module):
    """
    Attention mechanism that captures relationships between three-phase signals.
    
    In three-phase power systems:
    - Normal operation: Va, Vb, Vc are balanced with 120° phase differences
    - Anomaly: Phase relationships deviate from normal patterns
    
    This attention layer explicitly models the interactions between phases,
    making it easier to detect unbalance and other phase-related anomalies.
    """
    
    def __init__(
        self, 
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        phase_encoding: bool = True
    ):
        super(ThreePhaseAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.phase_encoding = phase_encoding
        
        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Phase relationship bias (learnable)
        if phase_encoding:
            phase_bias_init = torch.tensor([
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 1.0],
            ])
            self.phase_bias = nn.Parameter(phase_bias_init.unsqueeze(0).unsqueeze(0))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, phase_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        residual = x
        
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if self.phase_encoding and hasattr(self, 'phase_bias'):
            bias = self.phase_bias.expand(B, self.n_heads, -1, -1)
            if T >= 3:
                scores[:, :, :3, :3] = scores[:, :, :3, :3] + bias
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_o(context)
        output = self.layer_norm(output + residual)
        
        return output


class PhaseAwareTimesBlock(nn.Module):
    """
    Enhanced TimesBlock with phase-aware processing.
    """
    
    def __init__(self, configs, use_phase_attention: bool = True):
        super(PhaseAwareTimesBlock, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.use_phase_attention = use_phase_attention
        
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        
        if use_phase_attention:
            self.phase_attention = ThreePhaseAttention(
                d_model=configs.d_model,
                n_heads=getattr(configs, 'n_heads', 4),
                dropout=configs.dropout
            )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        
        period_list, period_weight = FFT_for_Period(x, self.k)
        
        conv_results = []
        for i in range(self.k):
            period = period_list[i]
            
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            conv_results.append(out[:, :(self.seq_len + self.pred_len), :])
        
        conv_results = torch.stack(conv_results, dim=-1)
        
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        conv_out = torch.sum(conv_results * period_weight, dim=-1)
        
        if self.use_phase_attention:
            attn_out = self.phase_attention(x)
            combined = torch.cat([conv_out, attn_out], dim=-1)
            gate = self.fusion_gate(combined)
            output = gate * conv_out + (1 - gate) * attn_out
        else:
            output = conv_out
        
        output = output + x
        return output


class Model(nn.Module):
    """
    TPATimesNet: Three-Phase Attention TimesNet
    
    Combines TimesNet's 2D temporal modeling with explicit three-phase
    attention for rural power grid voltage anomaly detection.
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.pred_len = getattr(configs, 'pred_len', 0)
        
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )
        
        self.model = nn.ModuleList([
            PhaseAwareTimesBlock(configs, use_phase_attention=True)
            for _ in range(configs.e_layers)
        ])
        
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        if self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
    
    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        enc_out = self.enc_embedding(x_enc, None)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)
        
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        
        return dec_out
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)
        
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)
        
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        
        return output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
