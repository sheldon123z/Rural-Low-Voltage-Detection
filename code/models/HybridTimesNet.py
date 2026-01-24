"""
HybridTimesNet: Hybrid Period Discovery TimesNet for Rural Voltage Anomaly Detection

Core Innovation: Combines data-driven FFT period discovery with domain-knowledge preset
periods specific to power systems. This hybrid approach ensures important electrical
periods are always captured while still allowing data-driven discovery.

Domain Knowledge for Power Systems:
- 20ms (1 cycle @ 50Hz): Fundamental electrical cycle
- 100ms (5 cycles): Short-term transients
- 1s (50 cycles): Sub-second variations
- 60s (1 minute): Minute-level patterns
- 900s (15 minutes): Quarter-hour settlement periods
- 3600s (1 hour): Hourly load patterns

Key Components:
1. Preset Period Module: Processes known electrical periods
2. FFT Discovery Module: Standard data-driven period discovery
3. Confidence-based Fusion: Combines preset and discovered periods based on reliability
4. Period Importance Scoring: Learns which periods matter most for anomaly detection

Author: Voltage Anomaly Detection Research
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


class PresetPeriodBlock(nn.Module):
    """Process data using preset domain-knowledge periods."""

    def __init__(self, configs, preset_periods: List[int]):
        super(PresetPeriodBlock, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, "pred_len", 0)
        valid_periods = [
            p for p in preset_periods if p >= 2 and p <= configs.seq_len // 2
        ]
        self.preset_periods = valid_periods if valid_periods else [2, 4, 8]

        self.period_convs = nn.ModuleDict()
        for p in self.preset_periods:
            self.period_convs[str(p)] = nn.Sequential(
                Inception_Block_V1(
                    configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
                ),
                nn.GELU(),
                Inception_Block_V1(
                    configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
                ),
            )

        self.importance_scores = nn.Parameter(torch.ones(len(self.preset_periods)))
        self.layer_norm = nn.LayerNorm(configs.d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle potential 4D input from embedding
        if x.dim() == 4:
            x = x.squeeze(-1) if x.size(-1) == 1 else x.mean(-1)
        B, T, N = x.size()
        total_len = self.seq_len + self.pred_len

        period_outputs = []

        for idx, period in enumerate(self.preset_periods):
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros([B, length - total_len, N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.period_convs[str(period)](out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            period_outputs.append(out[:, :total_len, :])

        weights = F.softmax(self.importance_scores, dim=0)
        period_outputs = torch.stack(period_outputs, dim=-1)
        weights = weights.view(1, 1, 1, -1)
        output = torch.sum(period_outputs * weights, dim=-1)

        output = self.layer_norm(output + x)

        return output, weights.squeeze()


class FFTDiscoveryBlock(nn.Module):
    """Standard FFT-based period discovery block."""

    def __init__(self, configs):
        super(FFTDiscoveryBlock, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, "pred_len", 0)
        self.k = configs.top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

        self.layer_norm = nn.LayerNorm(configs.d_model)

    def _fft_period_discovery(self, x: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        # Handle potential 4D input
        if x.dim() == 4:
            x = x.squeeze(-1) if x.size(-1) == 1 else x.mean(-1)
        B, T, C = x.size()

        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0

        k = min(self.k, len(frequency_list))
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()

        period_list = T / (top_list + 1e-8)
        period_list = np.clip(period_list.astype(int), 2, T // 2)

        return period_list, abs(xf).mean(-1)[:, top_list]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle potential 4D input from embedding
        if x.dim() == 4:
            x = x.squeeze(-1) if x.size(-1) == 1 else x.mean(-1)
        B, T, N = x.size()
        total_len = self.seq_len + self.pred_len

        period_list, period_weight = self._fft_period_discovery(x)

        res = []
        for i, period in enumerate(period_list):
            period = int(period)
            if period < 2:
                period = 2

            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros([B, length - total_len, N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        if len(res) == 0:
            return x, torch.ones(B, 1, device=x.device)

        res = torch.stack(res, dim=-1)

        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        output = torch.sum(res * period_weight, dim=-1)
        output = self.layer_norm(output + x)

        confidence = period_weight.mean(dim=(1, 2))

        return output, confidence


class ConfidenceBasedFusion(nn.Module):
    """Fuse preset and discovered periods based on confidence."""

    def __init__(self, d_model: int):
        super(ConfidenceBasedFusion, self).__init__()

        self.confidence_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2),
            nn.Softmax(dim=-1),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        preset_out: torch.Tensor,
        fft_out: torch.Tensor,
        preset_conf: torch.Tensor,
        fft_conf: torch.Tensor,
    ) -> torch.Tensor:
        combined = (preset_out + fft_out) / 2
        ctx = combined.mean(dim=1)
        weights = self.confidence_mlp(ctx)

        alpha = torch.sigmoid(self.alpha)
        final_weights = alpha * weights + (1 - alpha) * torch.tensor(
            [0.5, 0.5], device=weights.device
        )

        weights_expanded = final_weights.unsqueeze(1).unsqueeze(1)

        output = (
            weights_expanded[:, :, :, 0:1] * preset_out
            + weights_expanded[:, :, :, 1:2] * fft_out
        )

        return output


class HybridTimesBlock(nn.Module):
    """Combined hybrid block with preset and FFT discovery."""

    def __init__(self, configs, preset_periods: List[int]):
        super(HybridTimesBlock, self).__init__()

        self.preset_block = PresetPeriodBlock(configs, preset_periods)
        self.fft_block = FFTDiscoveryBlock(configs)
        self.fusion = ConfidenceBasedFusion(configs.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preset_out, preset_conf = self.preset_block(x)
        fft_out, fft_conf = self.fft_block(x)
        output = self.fusion(preset_out, fft_out, preset_conf, fft_conf)

        return output


class Model(nn.Module):
    """
    HybridTimesNet: Hybrid Period Discovery TimesNet

    Architecture:
    - Data Embedding
    - Stacked Hybrid TimesBlocks (each combines preset + FFT discovery)
    - Confidence-based Fusion
    - Task-specific Output

    Preset Periods (in sample points, adjust based on sampling rate):
    For typical 1Hz sampling: [50, 300, 900, 1800, 3600]
    For typical 0.1Hz sampling: [5, 30, 90, 180, 360]
    """

    DEFAULT_PRESET_PERIODS = [10, 20, 50, 100, 200]

    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, "label_len", 0)
        self.pred_len = getattr(configs, "pred_len", 0)

        preset_periods = getattr(configs, "preset_periods", self.DEFAULT_PRESET_PERIODS)
        valid_periods = [p for p in preset_periods if 2 <= p <= configs.seq_len // 2]
        if not valid_periods:
            valid_periods = self._auto_generate_periods(configs.seq_len)
        self.preset_periods = valid_periods

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.hybrid_blocks = nn.ModuleList(
            [
                HybridTimesBlock(configs, self.preset_periods)
                for _ in range(configs.e_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == "anomaly_detection" or self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def _auto_generate_periods(self, seq_len: int) -> List[int]:
        max_period = seq_len // 2
        periods = []
        p = 5
        while p <= max_period:
            periods.append(p)
            p = int(p * 2)
        if not periods:
            periods = [2, 4]
        return periods

    def _process_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.hybrid_blocks:
            x = block(x)
        return self.layer_norm(x)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._process_hybrid(enc_out)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.seq_len + self.pred_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.seq_len + self.pred_len, 1
        )

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out = self._process_hybrid(enc_out)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._process_hybrid(enc_out)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._process_hybrid(enc_out)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
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
