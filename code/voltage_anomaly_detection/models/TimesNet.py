"""
TimesNet Model for Voltage Anomaly Detection
Standalone version - independent from main TSLib

Paper: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
Link: https://openreview.net/pdf?id=ju_Uqw384Oq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    Use FFT to find the top-k periods in the time series.
    
    Args:
        x: Input tensor [B, T, C]
        k: Number of top periods to return
        
    Returns:
        period: List of periods
        period_weight: Amplitude weights [B, k]
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # Find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    TimesBlock: Core building block of TimesNet.
    
    Transforms 1D time series into 2D tensors based on discovered periods,
    applies 2D convolution, then reshapes back.
    """
    
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # Parameter-efficient design using Inception blocks
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # Padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # Reshape: 1D -> 2D
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: capture intra- and inter-period variations
            out = self.conv(out)
            # Reshape back: 2D -> 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        
        # Adaptive aggregation with softmax weights
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # Residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet Model for time series anomaly detection.
    
    Supports multiple tasks but optimized for anomaly detection in this standalone version.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.pred_len = getattr(configs, 'pred_len', 0)
        
        # Stack TimesBlocks
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        
        # Embedding layer
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, 
            configs.freq, configs.dropout
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        # Task-specific projection layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Forecasting task."""
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # TimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """Imputation task."""
        # Normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # TimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task.
        
        Uses reconstruction-based approach: the model learns to reconstruct
        normal patterns, and anomalies are detected by high reconstruction error.
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # Embedding (no temporal marks for anomaly detection)
        enc_out = self.enc_embedding(x_enc, None)
        
        # TimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Project back to original dimension
        dec_out = self.projection(enc_out)

        # De-Normalization
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """Classification task."""
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # TimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass with task routing.
        
        Args:
            x_enc: Input sequence [B, seq_len, enc_in]
            x_mark_enc: Time marks for encoder (can be None)
            x_dec: Decoder input (for forecast tasks)
            x_mark_dec: Time marks for decoder
            mask: Mask for imputation task
            
        Returns:
            Model output based on task_name
        """
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
