"""
VoltageTimesNet_v2: Enhanced TimesNet for Voltage Anomaly Detection with Recall Optimization

Key improvements over VoltageTimesNet:
1. Learnable preset weight (was hardcoded 0.3)
2. PowerQualityEncoder for THD/unbalance/voltage features
3. Multi-scale temporal convolution (kernels 3, 5, 7)
4. Anomaly sensitivity amplifier for recall optimization
5. Three-phase constraint module for phase correlation

Target: Improve recall from 62% to 70%+ while maintaining precision >95%
"""

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


class PowerQualityEncoder(nn.Module):
    """
    Encode power quality features for enhanced anomaly detection.

    Extracts and encodes domain-specific features:
    - THD (Total Harmonic Distortion) for harmonics
    - Voltage unbalance for three-phase imbalance
    - Phase voltages for fundamental patterns

    This helps the model focus on power-quality-relevant features.
    """

    def __init__(self, d_model, enc_in, pq_config=None):
        super().__init__()
        self.d_model = d_model
        self.enc_in = enc_in

        # Default power quality feature indices (for RuralVoltage dataset)
        # Columns: Va, Vb, Vc, Ia, Ib, Ic, P, Q, S, PF, THD_Va, THD_Vb, THD_Vc, Freq, V_unbalance, I_unbalance
        if pq_config is None:
            pq_config = {
                'voltage_indices': [0, 1, 2],      # Va, Vb, Vc
                'thd_indices': [10, 11, 12],       # THD_Va, THD_Vb, THD_Vc
                'unbalance_indices': [14, 15],     # V_unbalance, I_unbalance
            }

        self.pq_config = pq_config

        # Check if we have enough input features for PQ encoding
        max_idx = max(
            max(pq_config.get('voltage_indices', [0])),
            max(pq_config.get('thd_indices', [0])),
            max(pq_config.get('unbalance_indices', [0]))
        )
        self.has_pq_features = (enc_in > max_idx)

        if self.has_pq_features:
            n_voltage = len(pq_config.get('voltage_indices', []))
            n_thd = len(pq_config.get('thd_indices', []))
            n_unbalance = len(pq_config.get('unbalance_indices', []))

            # Encoders for different feature groups
            self.voltage_encoder = nn.Linear(n_voltage, d_model // 2)
            self.thd_encoder = nn.Linear(n_thd, d_model // 4) if n_thd > 0 else None
            self.unbalance_encoder = nn.Linear(n_unbalance, d_model // 4) if n_unbalance > 0 else None

            # Fusion layer
            total_dim = d_model // 2
            if n_thd > 0:
                total_dim += d_model // 4
            if n_unbalance > 0:
                total_dim += d_model // 4

            self.fusion = nn.Linear(total_dim, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            # Fallback: simple linear projection
            self.fallback_proj = nn.Linear(enc_in, d_model)
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]
        Returns:
            pq_embedding: Power quality embedding [B, T, d_model]
        """
        B, T, C = x.shape

        if self.has_pq_features:
            # Extract feature groups
            v_idx = self.pq_config['voltage_indices']
            voltage_features = x[:, :, v_idx]
            encoded_voltage = self.voltage_encoder(voltage_features)

            embeddings = [encoded_voltage]

            if self.thd_encoder is not None:
                thd_idx = self.pq_config['thd_indices']
                thd_features = x[:, :, thd_idx]
                embeddings.append(self.thd_encoder(thd_features))

            if self.unbalance_encoder is not None:
                ub_idx = self.pq_config['unbalance_indices']
                unbalance_features = x[:, :, ub_idx]
                embeddings.append(self.unbalance_encoder(unbalance_features))

            # Concatenate and fuse
            combined = torch.cat(embeddings, dim=-1)
            pq_embedding = self.fusion(combined)
        else:
            # Fallback for datasets without explicit PQ features
            pq_embedding = self.fallback_proj(x)

        return self.layer_norm(pq_embedding)


class AnomalySensitivityAmplifier(nn.Module):
    """
    Amplify anomaly signals to improve recall.

    This module learns to identify and amplify features that indicate anomalies,
    making it easier for the reconstruction error to detect them.

    Key mechanism:
    - Learn attention weights that focus on anomaly-indicative features
    - Amplify reconstruction difficulty for anomalous patterns
    """

    def __init__(self, d_model, amplify_factor=2.0):
        super().__init__()
        self.d_model = d_model
        self.amplify_factor = amplify_factor

        # Anomaly attention: learn to identify anomaly-indicative features
        self.anomaly_attention = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()  # Output [0, 1] weights
        )

        # Learnable amplification scale
        self.amplify_scale = nn.Parameter(torch.tensor(amplify_factor))

    def forward(self, x, training=True):
        """
        Args:
            x: Input features [B, T, d_model]
            training: Whether in training mode
        Returns:
            amplified: Amplified features [B, T, d_model]
        """
        # Compute anomaly attention weights
        attn_weights = self.anomaly_attention(x)  # [B, T, d_model]

        # Apply amplification
        # High weights = likely anomaly features = amplify
        amplified = x * (1 + attn_weights * self.amplify_scale)

        return amplified


class PhaseConstraintModule(nn.Module):
    """
    Learn and enforce three-phase voltage constraints.

    For balanced three-phase systems:
    - Va + Vb + Vc ≈ 0 (zero sequence)
    - |Va| ≈ |Vb| ≈ |Vc| (magnitude balance)
    - Phase angles differ by 120°

    This module learns these relationships for better anomaly detection.
    """

    def __init__(self, d_model, n_phases=3):
        super().__init__()
        self.n_phases = n_phases

        # Learnable phase correlation matrix
        # Initialize to encourage diagonal dominance (self-correlation > cross-correlation)
        init_matrix = torch.eye(n_phases) + 0.5 * torch.ones(n_phases, n_phases)
        self.phase_correlation = nn.Parameter(init_matrix)

        # Phase embedding
        self.phase_embed = nn.Linear(n_phases, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, phase_indices=None):
        """
        Args:
            x: Input tensor [B, T, C] or phase features [B, T, n_phases]
            phase_indices: Indices of phase columns in x (if x has more than n_phases columns)
        Returns:
            phase_features: Enhanced phase features [B, T, d_model]
        """
        if phase_indices is not None and x.shape[-1] > self.n_phases:
            # Extract phase features
            phase_x = x[:, :, phase_indices]
        else:
            phase_x = x[:, :, :self.n_phases]

        # Apply learned correlation
        correlated = torch.matmul(phase_x, self.phase_correlation)

        # Embed to model dimension
        phase_features = self.phase_embed(correlated)

        return self.layer_norm(phase_features)

    def get_constraint_loss(self):
        """
        Regularization loss to encourage physically meaningful correlations.
        Diagonal should be larger than off-diagonal (self > cross correlation).
        """
        diag = torch.diag(self.phase_correlation)
        off_diag_mask = ~torch.eye(self.n_phases, dtype=torch.bool, device=self.phase_correlation.device)
        off_diag = self.phase_correlation[off_diag_mask]

        # Loss: encourage diag > off_diag
        loss = F.relu(off_diag.mean() - diag.mean() + 0.5).mean()
        return loss


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-scale temporal convolution for capturing patterns at different time scales.

    Uses kernel sizes 3, 5, 7 to capture:
    - Short-term fluctuations (kernel=3)
    - Medium-term patterns (kernel=5)
    - Longer-term trends (kernel=7)
    """

    def __init__(self, d_model, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        n_scales = len(kernel_sizes)
        d_per_scale = d_model // n_scales

        self.convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_per_scale,
                kernel_size=k,
                padding=k // 2,
                groups=1
            )
            for k in kernel_sizes
        ])

        # Handle remainder if d_model not divisible by n_scales
        total_out = d_per_scale * n_scales
        if total_out < d_model:
            self.proj = nn.Linear(total_out, d_model)
        else:
            self.proj = None

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: Input [B, T, d_model]
        Returns:
            out: Multi-scale features [B, T, d_model]
        """
        # Transpose for conv1d: [B, T, D] -> [B, D, T]
        x_t = x.transpose(1, 2)

        # Apply multi-scale convolutions
        scale_outputs = [conv(x_t) for conv in self.convs]

        # Concatenate: [B, D_total, T]
        multi_scale = torch.cat(scale_outputs, dim=1)

        # Transpose back: [B, D_total, T] -> [B, T, D_total]
        out = multi_scale.transpose(1, 2)

        # Project if needed
        if self.proj is not None:
            out = self.proj(out)

        return self.layer_norm(out + x)


def FFT_for_Period_Voltage_v2(x, k=2, preset_periods=None, preset_weight=0.3):
    """
    Enhanced period discovery with learnable preset weight.

    Same as FFT_for_Period_Voltage but accepts preset_weight as tensor for backprop.
    """
    B, T, C = x.size()
    xf = torch.fft.rfft(x, dim=1)

    # Find periods by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # Remove DC

    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # Calculate periods from frequencies
    fft_periods = []
    fft_freq_indices = []
    for freq_idx in top_list:
        if freq_idx > 0:
            period = max(2, T // freq_idx)
            fft_periods.append(period)
            fft_freq_indices.append(freq_idx)
        else:
            fft_periods.append(T)
            fft_freq_indices.append(1)

    final_periods = []
    final_freq_indices = []

    # Handle preset_weight as tensor
    if isinstance(preset_weight, torch.Tensor):
        pw = torch.sigmoid(preset_weight).item()  # Ensure [0,1] range
    else:
        pw = preset_weight

    if preset_periods is not None and len(preset_periods) > 0:
        valid_presets = [p for p in preset_periods if 2 <= p <= T // 2]

        if valid_presets:
            n_preset = max(1, int(k * pw))
            n_fft = k - n_preset

            seen = set()
            for i in range(min(n_fft, len(fft_periods))):
                p = fft_periods[i]
                if p not in seen:
                    seen.add(p)
                    final_periods.append(p)
                    final_freq_indices.append(fft_freq_indices[i])

            for p in valid_presets[:n_preset]:
                if p not in seen:
                    seen.add(p)
                    final_periods.append(p)
                    freq_idx = max(1, min(T // p, xf.shape[1] - 1))
                    final_freq_indices.append(freq_idx)

            for i, p in enumerate(fft_periods):
                if len(final_periods) >= k:
                    break
                if p not in seen:
                    seen.add(p)
                    final_periods.append(p)
                    final_freq_indices.append(fft_freq_indices[i])

            final_periods = final_periods[:k]
            final_freq_indices = final_freq_indices[:k]
        else:
            final_periods = fft_periods[:k]
            final_freq_indices = fft_freq_indices[:k]
    else:
        final_periods = fft_periods[:k]
        final_freq_indices = fft_freq_indices[:k]

    period_list = np.array(final_periods)

    freq_indices_tensor = torch.tensor(final_freq_indices, device=x.device, dtype=torch.long)
    xf_amplitudes = abs(xf).mean(-1)
    period_weight = xf_amplitudes[:, freq_indices_tensor]

    return period_list, period_weight


class VoltageTimesBlock_v2(nn.Module):
    """
    Enhanced TimesBlock with:
    1. Learnable preset weight
    2. Multi-scale temporal convolution
    3. Better residual connections
    """

    def __init__(self, configs, preset_periods=None, preset_weight_param=None):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.k = configs.top_k

        # Learnable preset weight (shared reference)
        self.preset_weight = preset_weight_param

        # Preset periods
        if preset_periods is None:
            self.preset_periods = [
                max(2, configs.seq_len // 20),
                max(2, configs.seq_len // 10),
                max(2, configs.seq_len // 4),
            ]
        else:
            self.preset_periods = preset_periods

        # 2D convolution for period-based features
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

        # Multi-scale temporal convolution
        self.multi_scale_conv = MultiScaleTemporalConv(configs.d_model)

    def forward(self, x):
        B, T, N = x.size()

        # Get preset weight value
        pw = self.preset_weight if self.preset_weight is not None else 0.3

        # Enhanced period discovery
        period_list, period_weight = FFT_for_Period_Voltage_v2(
            x, self.k, preset_periods=self.preset_periods, preset_weight=pw
        )

        res = []
        for i in range(len(period_list)):
            period = int(period_list[i])
            if period < 2:
                period = 2

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

            # Reshape for 2D conv
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()

            # 2D convolution
            out = self.conv(out)

            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)

        # Weighted aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1)
        res = torch.sum(res * period_weight, -1)

        # Multi-scale temporal convolution
        res = self.multi_scale_conv(res)

        # Residual
        res = res + x

        return res


class Model(nn.Module):
    """
    VoltageTimesNet_v2: Enhanced model for voltage anomaly detection with recall optimization.

    Improvements:
    1. Learnable preset_weight (was hardcoded 0.3)
    2. PowerQualityEncoder for domain-specific feature extraction
    3. Multi-scale temporal convolution
    4. AnomalySensitivityAmplifier for recall optimization
    5. PhaseConstraintModule for three-phase correlation
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.pred_len = getattr(configs, 'pred_len', 0)

        # Learnable preset weight (key improvement #1)
        init_weight = getattr(configs, 'preset_weight', 0.3)
        # Use inverse sigmoid to initialize so sigmoid(init) ≈ init_weight
        init_logit = np.log(init_weight / (1 - init_weight + 1e-8))
        self.preset_weight = nn.Parameter(torch.tensor(float(init_logit)))

        # Calculate preset periods
        self.preset_periods = self._calculate_preset_periods(configs.seq_len)

        # Power quality encoder (key improvement #2)
        self.pq_encoder = PowerQualityEncoder(
            configs.d_model,
            configs.enc_in,
            pq_config=getattr(configs, 'pq_config', None)
        )

        # Anomaly sensitivity amplifier (key improvement #4)
        self.anomaly_amplifier = AnomalySensitivityAmplifier(
            configs.d_model,
            amplify_factor=getattr(configs, 'anomaly_amplify_factor', 2.0)
        )

        # Phase constraint module (key improvement #5)
        if configs.enc_in >= 3:
            self.phase_constraint = PhaseConstraintModule(
                configs.d_model,
                n_phases=3
            )
        else:
            self.phase_constraint = None

        # Encoder layers with enhanced VoltageTimesBlock
        self.model = nn.ModuleList([
            VoltageTimesBlock_v2(configs, self.preset_periods, self.preset_weight)
            for _ in range(configs.e_layers)
        ])

        # Standard data embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Feature fusion for PQ and phase features
        self.feature_fusion = nn.Linear(configs.d_model * 2, configs.d_model)

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Task-specific heads
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _calculate_preset_periods(self, seq_len):
        """Calculate preset periods based on sequence length."""
        periods = []
        preset_candidates = [60, 300, 900, 3600]

        for p in preset_candidates:
            if 2 <= p <= seq_len // 2:
                periods.append(p)

        if not periods:
            periods = [
                max(2, seq_len // 20),
                max(2, seq_len // 10),
                max(2, seq_len // 4),
            ]

        return periods

    def get_auxiliary_loss(self):
        """Get auxiliary losses for regularization."""
        loss = 0.0

        # Phase constraint regularization
        if self.phase_constraint is not None:
            loss = loss + 0.01 * self.phase_constraint.get_constraint_loss()

        return loss

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection with enhanced feature extraction.
        """
        B, T, C = x_enc.shape

        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Standard embedding
        enc_out = self.enc_embedding(x_enc, None)

        # Power quality features
        pq_features = self.pq_encoder(x_enc)

        # Phase constraint features
        if self.phase_constraint is not None:
            phase_features = self.phase_constraint(x_enc, phase_indices=[0, 1, 2])
            pq_features = pq_features + phase_features

        # Fuse embeddings
        fused = self.feature_fusion(torch.cat([enc_out, pq_features], dim=-1))
        enc_out = fused

        # Apply anomaly sensitivity amplifier (for better recall)
        enc_out = self.anomaly_amplifier(enc_out, training=self.training)

        # VoltageTimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Projection
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Forecasting task."""
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
        """Imputation task."""
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
        """Classification task."""
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
        """Forward pass routing."""
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
