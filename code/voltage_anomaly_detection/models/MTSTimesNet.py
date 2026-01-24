"""
MTSTimesNet: Multi-scale Temporal TimesNet for Rural Voltage Anomaly Detection

Core Innovation: Parallel multi-scale temporal branches that simultaneously capture
patterns at different time scales (short-term fluctuations, medium-term trends, long-term patterns).

Rural power grids exhibit multi-scale temporal patterns:
- Short-term (seconds to minutes): Transient events, voltage sags
- Medium-term (minutes to hours): Load variations, daily patterns
- Long-term (hours to days): Seasonal patterns, systematic issues

Key Components:
1. Multi-scale TimesBlocks: Parallel branches with different period focus
2. Adaptive Fusion Gate: Learns optimal combination of scales
3. Cross-scale Residual Connections: Information flow across scales

Author: Voltage Anomaly Detection Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from typing import List, Tuple, Optional

from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period_Range(x, k=2, min_period=2, max_period=None):
    """FFT-based period discovery with range constraints."""
    B, T, C = x.size()
    if max_period is None:
        max_period = T // 2

    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0

    periods = T / (torch.arange(len(frequency_list), device=x.device) + 1e-8)
    mask = (periods >= min_period) & (periods <= max_period)
    frequency_list = frequency_list * mask.float()

    _, top_list = torch.topk(frequency_list, min(k, mask.sum().item()))
    top_list = top_list.detach().cpu().numpy()

    period_list = T // (top_list + 1)
    period_list = np.clip(period_list, min_period, max_period)

    return period_list, abs(xf).mean(-1)[:, top_list]


class ScaleSpecificTimesBlock(nn.Module):
    """TimesBlock focused on a specific temporal scale."""

    def __init__(
        self,
        configs,
        scale_name: str = "medium",
        min_period: int = 10,
        max_period: int = 50,
    ):
        super(ScaleSpecificTimesBlock, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.scale_name = scale_name
        self.min_period = min_period
        self.max_period = min(max_period, configs.seq_len // 2)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N = x.size()

        period_list, period_weight = FFT_for_Period_Range(
            x, self.k, self.min_period, self.max_period
        )

        res = []
        for i in range(len(period_list)):
            period = int(period_list[i])
            if period < 2:
                period = 2

            total_len = self.seq_len + self.pred_len
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

        return output, period_weight.mean(dim=(1, 2))


class AdaptiveFusionGate(nn.Module):
    """Adaptive gate for fusing multi-scale features."""

    def __init__(self, d_model: int, n_scales: int = 3):
        super(AdaptiveFusionGate, self).__init__()

        self.n_scales = n_scales
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.gate_network = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_scales),
            nn.Softmax(dim=-1),
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        B, T, D = scale_features[0].size()

        contexts = []
        for feat in scale_features:
            ctx = self.global_pool(feat.transpose(1, 2)).squeeze(-1)
            contexts.append(ctx)

        combined_ctx = torch.cat(contexts, dim=-1)
        weights = self.gate_network(combined_ctx)

        weights = weights.unsqueeze(1).unsqueeze(-1)
        stacked = torch.stack(scale_features, dim=2)
        fused = (stacked * weights).sum(dim=2)

        return fused


class CrossScaleConnection(nn.Module):
    """Cross-scale residual connections for information exchange between scales."""

    def __init__(self, d_model: int, n_scales: int = 3):
        super(CrossScaleConnection, self).__init__()

        self.n_scales = n_scales
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=0.1, batch_first=True
        )
        self.projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_scales)]
        )

    def forward(self, scale_features: List[torch.Tensor]) -> List[torch.Tensor]:
        B, T, D = scale_features[0].size()
        all_scales = torch.cat(scale_features, dim=1)

        enhanced = []
        for i, feat in enumerate(scale_features):
            attended, _ = self.cross_attention(feat, all_scales, all_scales)
            enhanced_feat = self.projections[i](feat + attended)
            enhanced.append(enhanced_feat)

        return enhanced


class Model(nn.Module):
    """
    MTSTimesNet: Multi-scale Temporal TimesNet

    Architecture:
    - Shared Embedding Layer
    - Parallel Multi-scale TimesBlocks
    - Cross-scale Connections
    - Adaptive Fusion Gate
    - Output Projection
    """

    SCALE_CONFIGS = {
        "short": {"min_period": 2, "max_period": 20},
        "medium": {"min_period": 20, "max_period": 60},
        "long": {"min_period": 60, "max_period": 200},
    }

    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, "label_len", 0)
        self.pred_len = getattr(configs, "pred_len", 0)

        self._adjust_scale_configs()

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.n_layers = configs.e_layers
        self.n_scales = len(self.SCALE_CONFIGS)

        self.scale_blocks = nn.ModuleDict()
        for scale_name, scale_cfg in self.SCALE_CONFIGS.items():
            self.scale_blocks[scale_name] = nn.ModuleList(
                [
                    ScaleSpecificTimesBlock(
                        configs,
                        scale_name=scale_name,
                        min_period=scale_cfg["min_period"],
                        max_period=scale_cfg["max_period"],
                    )
                    for _ in range(self.n_layers)
                ]
            )

        self.cross_scale = nn.ModuleList(
            [
                CrossScaleConnection(configs.d_model, self.n_scales)
                for _ in range(self.n_layers - 1)
            ]
        )

        self.fusion_gate = AdaptiveFusionGate(configs.d_model, self.n_scales)
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

    def _adjust_scale_configs(self):
        seq_len = self.seq_len
        for scale_name in self.SCALE_CONFIGS:
            max_period = self.SCALE_CONFIGS[scale_name]["max_period"]
            if max_period > seq_len // 2:
                self.SCALE_CONFIGS[scale_name]["max_period"] = seq_len // 2

        valid_scales = {
            k: v
            for k, v in self.SCALE_CONFIGS.items()
            if v["min_period"] < v["max_period"]
        }

        if len(valid_scales) < 3:
            self.SCALE_CONFIGS = {
                "short": {"min_period": 2, "max_period": max(4, seq_len // 10)},
                "medium": {
                    "min_period": max(4, seq_len // 10),
                    "max_period": max(8, seq_len // 5),
                },
                "long": {
                    "min_period": max(8, seq_len // 5),
                    "max_period": seq_len // 2,
                },
            }

    def _process_multi_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale_names = list(self.scale_blocks.keys())
        scale_features = {name: x for name in scale_names}

        for layer_idx in range(self.n_layers):
            new_features = {}
            for scale_name in scale_names:
                # self.scale_blocks[scale_name] 是 ModuleList，直接索引
                block_list = self.scale_blocks[scale_name]
                feat, _ = block_list[layer_idx](scale_features[scale_name])
                new_features[scale_name] = feat

            if layer_idx < self.n_layers - 1:
                feature_list = [new_features[name] for name in scale_names]
                enhanced_list = self.cross_scale[layer_idx](feature_list)
                for i, name in enumerate(scale_names):
                    new_features[name] = enhanced_list[i]

            scale_features = new_features

        feature_list = [scale_features[name] for name in scale_names]
        fused = self.fusion_gate(feature_list)

        return self.layer_norm(fused)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._process_multi_scale(enc_out)
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
        enc_out = self._process_multi_scale(enc_out)
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
        enc_out = self._process_multi_scale(enc_out)
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
        enc_out = self._process_multi_scale(enc_out)

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
