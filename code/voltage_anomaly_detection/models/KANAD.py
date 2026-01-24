"""
KANAD Model for Anomaly Detection
Kolmogorov-Arnold Network for Anomaly Detection
Standalone version - independent from main TSLib
"""

import numpy as np
import torch
import torch.nn as nn


def rearrange_bd_to_batch(x, B):
    """Rearrange from (B*D, L) to (B, L, D)."""
    D = x.shape[0] // B
    L = x.shape[1]
    return x.view(B, D, L).permute(0, 2, 1)


def rearrange_batch_to_bd(x):
    """Rearrange from (B, L, D) to (B*D, L)."""
    B, L, D = x.shape
    return x.permute(0, 2, 1).reshape(B * D, L)


class KANADModel(nn.Module):
    """Kolmogorov-Arnold Network for Anomaly Detection core module."""
    
    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.order = order
        self.window = window
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(self.window, self.order).unsqueeze(0),
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(window, window)

    def forward(self, x: torch.Tensor, return_last: bool = False, *args, **kwargs):
        res = []
        res.append(x.unsqueeze(1))
        ff = torch.concat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [torch.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        if return_last:
            return ff.squeeze(1), ff
        return ff.squeeze(1)

    def _create_custom_periodic_cosine(self, window: int, period) -> torch.Tensor:
        d = len(period) if isinstance(period, list) else period
        pl = period if isinstance(period, list) else [i for i in range(1, period + 1)]
        result = torch.empty(d, window, dtype=torch.float32)
        for i, p in enumerate(pl):
            t = torch.arange(0, 1, 1 / window, dtype=torch.float32) / p * 2 * np.pi
            result[i, :] = torch.cos(t)
        return result


class Model(nn.Module):
    """
    KANAD: Kolmogorov-Arnold Network for Anomaly Detection
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.pred_len = getattr(configs, 'pred_len', configs.seq_len)
        self.order = configs.d_model

        # Encoder
        self.enc = KANADModel(window=self.seq_len, order=configs.d_model)

    def anomaly_detection(self, x_enc):
        # Reshape the input [B, L, D] to [B * D, L]
        x_input = rearrange_batch_to_bd(x_enc)
        enc_out = self.enc(x_input)
        # [B * D, L] -> [B, L, D]
        dec_out = rearrange_bd_to_batch(enc_out, x_enc.size(0))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            raise NotImplementedError("Task forecasting for KANAD is not supported")
        if self.task_name == "imputation":
            raise NotImplementedError("Task imputation for KANAD is not supported")
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == "classification":
            raise NotImplementedError("Task classification for KANAD is not supported")
        return None
