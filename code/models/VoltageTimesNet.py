"""
VoltageTimesNet: TimesNet variant optimized for Rural Power Grid Voltage Anomaly Detection

Key innovations:
1. Preset Period Discovery: Combines FFT-discovered periods with power grid preset periods
   (50Hz power frequency, daily load pattern, weekly pattern)
2. Voltage-specific TimesBlock: Enhanced 2D convolution for voltage signal patterns
3. Three-phase aware processing: Designed for Va, Vb, Vc correlations

Paper reference: Adapted from TimesNet (ICLR 2023)
"""

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< HEAD
=======
from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding

>>>>>>> origin/copilot/optimize-ci-cd-workflow
# Power grid preset periods (in samples, assuming 1-second sampling rate)
VOLTAGE_PRESET_PERIODS = {
    "1min": 60,  # 1 minute cycle for short-term fluctuations
    "5min": 300,  # 5 minute cycle for transient events
    "15min": 900,  # 15 minute cycle for load variations
    "1h": 3600,  # 1 hour cycle for hourly load pattern
}


def FFT_for_Period_Voltage(x, k=2, preset_periods=None, preset_weight=0.3):
    """
    Enhanced period discovery for voltage signals.

    Combines FFT-based period discovery with domain knowledge of power grid patterns.
    For voltage signals, we know certain periods are important:
    - Power frequency harmonics (2nd, 3rd, 5th, 7th...)
    - Daily load cycles
    - Weekly patterns

    Args:
        x: Input tensor [B, T, C]
        k: Number of top periods to select
        preset_periods: List of preset period values to consider
        preset_weight: Weight for preset periods (0-1)

    Returns:
        period_list: List of selected periods
        period_weight: Weights for each period
    """
    # [B, T, C]
    B, T, C = x.size()
    xf = torch.fft.rfft(x, dim=1)

    # Find periods by amplitudes (FFT-based discovery)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # Remove DC component

    # Get top-k FFT-discovered periods
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # Calculate periods from frequencies
    fft_periods = []
    for freq_idx in top_list:
        if freq_idx > 0:
            period = max(2, T // freq_idx)  # Ensure period >= 2
            fft_periods.append(period)
        else:
            fft_periods.append(T)

    # Combine with preset periods if provided
    if preset_periods is not None and len(preset_periods) > 0:
        # Filter preset periods that fit within sequence length
        valid_presets = [p for p in preset_periods if 2 <= p <= T // 2]

        if valid_presets:
            # Calculate combined period list
            # Take some from FFT and some from presets based on weight
            n_preset = max(1, int(k * preset_weight))
            n_fft = k - n_preset

            combined_periods = fft_periods[:n_fft] + valid_presets[:n_preset]

            # Remove duplicates while preserving order
            seen = set()
            unique_periods = []
            for p in combined_periods:
                if p not in seen:
                    seen.add(p)
                    unique_periods.append(p)

            # Pad with FFT periods if needed
            while len(unique_periods) < k:
                for p in fft_periods:
                    if p not in seen and len(unique_periods) < k:
                        seen.add(p)
                        unique_periods.append(p)

            period_list = np.array(unique_periods[:k])
        else:
            period_list = np.array(fft_periods[:k])
    else:
        period_list = np.array(fft_periods[:k])

    # Calculate period weights based on FFT amplitudes
    period_weight = abs(xf).mean(-1)[:, top_list[: len(period_list)]]

    return period_list, period_weight


class VoltageTimesBlock(nn.Module):
    """
    Enhanced TimesBlock for voltage signal processing.

    Key differences from standard TimesBlock:
    1. Uses preset periods in addition to FFT-discovered periods
    2. Multi-scale convolution kernels optimized for power signals
    3. Residual connections for better gradient flow
    """

    def __init__(self, configs, preset_periods=None):
        super(VoltageTimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # Preset periods for voltage signals
        if preset_periods is None:
            # Default: derive from sequence length
            self.preset_periods = [
                max(2, configs.seq_len // 20),  # ~5% of sequence
                max(2, configs.seq_len // 10),  # ~10% of sequence
                max(2, configs.seq_len // 4),  # ~25% of sequence
            ]
        else:
            self.preset_periods = preset_periods

        # 2D convolution for period-based feature extraction
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

        # Optional: Additional 1D conv for temporal smoothing
        self.temporal_conv = nn.Conv1d(
            configs.d_model,
            configs.d_model,
            kernel_size=3,
            padding=1,
            groups=configs.d_model,
        )

    def forward(self, x):
        B, T, N = x.size()

        # Enhanced period discovery with preset periods
        period_list, period_weight = FFT_for_Period_Voltage(
            x, self.k, preset_periods=self.preset_periods, preset_weight=0.3
        )

        res = []
        for i in range(len(period_list)):
            period = int(period_list[i])

            # Ensure period is valid
            if period < 2:
                period = 2

            # Padding to make sequence length divisible by period
            total_len = self.seq_len + self.pred_len
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], length - total_len, x.shape[2]], device=x.device
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            # Reshape for 2D convolution: [B, T, N] -> [B, N, T//period, period]
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()

            # Apply 2D convolution
            out = self.conv(out)

            # Reshape back: [B, N, T//period, period] -> [B, T, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)

        # Adaptive aggregation with softmax weights
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)
        period_weight = period_weight.repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # Apply temporal smoothing
        res = res + self.temporal_conv(res.permute(0, 2, 1)).permute(0, 2, 1)

        # Residual connection
        res = res + x

        return res


class Model(nn.Module):
    """
    VoltageTimesNet: TimesNet variant for Rural Power Grid Voltage Anomaly Detection

    Designed for time series anomaly detection on voltage data with features:
    - Three-phase voltage (Va, Vb, Vc)
    - Three-phase current (Ia, Ib, Ic)
    - Power metrics (P, Q, S, PF)
    - Power quality metrics (THD, Freq, Unbalance)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, "label_len", 0)
        self.pred_len = getattr(configs, "pred_len", 0)

        # Calculate preset periods based on sequence length
        self.preset_periods = self._calculate_preset_periods(configs.seq_len)

        # Encoder layers with VoltageTimesBlock
        self.model = nn.ModuleList(
            [
                VoltageTimesBlock(configs, self.preset_periods)
                for _ in range(configs.e_layers)
            ]
        )

        # Data embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Task-specific heads
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def _calculate_preset_periods(self, seq_len):
        """
        Calculate preset periods based on sequence length and power grid knowledge.

        For voltage monitoring data sampled at 1Hz:
        - 1 minute cycle (60 samples) for short-term variations
        - 5 minute cycle (300 samples) for transient events
        - 15 minute cycle (900 samples) for load variations
        """
        periods = []

        # Common preset periods for power grid data
        preset_candidates = [60, 300, 900, 3600]  # 1min, 5min, 15min, 1h

        for p in preset_candidates:
            if 2 <= p <= seq_len // 2:
                periods.append(p)

        # Add sequence-relative periods if no absolute presets fit
        if not periods:
            periods = [
                max(2, seq_len // 20),
                max(2, seq_len // 10),
                max(2, seq_len // 4),
            ]

        return periods

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Forecasting task."""
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # VoltageTimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Projection
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """Imputation task."""
        # Normalization with mask
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # VoltageTimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Projection
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )

        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task.

        For voltage data, the model learns to reconstruct normal patterns.
        High reconstruction error indicates anomalies.
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Embedding (no temporal marks for anomaly detection)
        enc_out = self.enc_embedding(x_enc, None)

        # VoltageTimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Projection
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """Classification task."""
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)

        # VoltageTimesNet layers
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
        """Forward pass routing to task-specific methods."""
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
