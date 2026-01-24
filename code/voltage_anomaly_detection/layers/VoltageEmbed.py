"""
VoltageEmbed: Specialized Embedding Layers for Power Grid Voltage Signals

This module provides domain-specific embeddings designed for rural power grid
voltage anomaly detection:

1. PowerFrequencyEmbedding: Encodes 50Hz power frequency cycles
2. DailyLoadEmbedding: Captures daily load patterns (24-hour cycles)
3. ThreePhaseEmbedding: Encodes Va-Vb-Vc phase relationships (120-degree shift)
4. VoltageDataEmbedding: Combined embedding for voltage signals

Key innovations:
- Exploit known periodicities in power systems (50Hz, daily, weekly)
- Encode three-phase relationships (phase angle differences)
- Integrate voltage quality features (THD, unbalance) into embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PowerFrequencyEmbedding(nn.Module):
    """
    Embedding that encodes power frequency cycle information.

    In 50Hz power systems, each cycle is 20ms. For data sampled at different
    rates, this embedding helps the model understand where each sample falls
    within the power frequency cycle.

    For anomaly detection with second-level sampling, this embedding encodes
    the phase relationship with respect to longer-term harmonics.
    """

    def __init__(self, d_model, max_len=5000, power_freq=50.0, sample_rate=1.0):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            power_freq: Power system frequency (50Hz or 60Hz)
            sample_rate: Sampling rate in Hz
        """
        super(PowerFrequencyEmbedding, self).__init__()

        self.d_model = d_model
        self.power_freq = power_freq
        self.sample_rate = sample_rate

        # Pre-compute power frequency cycle embeddings
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)

        # Multiple harmonics of power frequency
        harmonics = [1, 2, 3, 5, 7]  # Fundamental + common harmonics
        harmonic_dim = d_model // (len(harmonics) * 2)

        for h_idx, harmonic in enumerate(harmonics):
            freq = power_freq * harmonic
            # Angular frequency considering sample rate
            omega = 2.0 * math.pi * freq / sample_rate

            start_idx = h_idx * harmonic_dim * 2
            end_idx = min(start_idx + harmonic_dim * 2, d_model)

            # Use alternating sin/cos for each harmonic
            for i in range(0, end_idx - start_idx, 2):
                phase_shift = i * math.pi / (end_idx - start_idx)
                if start_idx + i < d_model:
                    pe[:, start_idx + i] = torch.sin(position.squeeze() * omega + phase_shift)
                if start_idx + i + 1 < d_model:
                    pe[:, start_idx + i + 1] = torch.cos(position.squeeze() * omega + phase_shift)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]

        Returns:
            Power frequency positional encoding [B, T, d_model]
        """
        return self.pe[:, :x.size(1)]


class DailyLoadEmbedding(nn.Module):
    """
    Embedding that captures daily load patterns in power systems.

    Power consumption follows predictable daily patterns:
    - Morning peak (7-9 AM)
    - Midday trough (12-2 PM)
    - Evening peak (6-9 PM)
    - Night low (12-6 AM)

    This embedding helps the model understand time-of-day context.
    """

    def __init__(self, d_model, samples_per_day=86400):
        """
        Args:
            d_model: Embedding dimension
            samples_per_day: Number of samples per day (86400 for 1Hz sampling)
        """
        super(DailyLoadEmbedding, self).__init__()

        self.d_model = d_model
        self.samples_per_day = samples_per_day

        # Pre-compute daily cycle embeddings
        # Use multiple frequencies to capture different daily patterns
        daily_periods = [
            samples_per_day,          # Full day cycle
            samples_per_day // 2,     # Half-day (AM/PM)
            samples_per_day // 3,     # 8-hour cycles
            samples_per_day // 4,     # 6-hour cycles
            samples_per_day // 6,     # 4-hour cycles
        ]

        self.period_embeddings = nn.ModuleList([
            nn.Embedding(max(period, 1), d_model // len(daily_periods))
            for period in daily_periods
        ])

        # Projection to combine period embeddings
        self.projection = nn.Linear(d_model // len(daily_periods) * len(daily_periods), d_model)

    def forward(self, x, time_indices=None):
        """
        Args:
            x: Input tensor [B, T, C]
            time_indices: Optional time indices within day [B, T]

        Returns:
            Daily load pattern embedding [B, T, d_model]
        """
        B, T, _ = x.size()

        if time_indices is None:
            # Assume sequential time indices
            time_indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        embeddings = []
        for i, emb in enumerate(self.period_embeddings):
            period = emb.num_embeddings
            period_idx = (time_indices % period).long()
            embeddings.append(emb(period_idx))

        combined = torch.cat(embeddings, dim=-1)
        return self.projection(combined)


class ThreePhaseEmbedding(nn.Module):
    """
    Embedding that encodes three-phase relationships for Va, Vb, Vc.

    In balanced three-phase systems:
    - Va, Vb, Vc are 120 degrees (2π/3) apart
    - Positive sequence: Va leads Vb leads Vc
    - Negative sequence: Va leads Vc leads Vb (indicates unbalance)

    This embedding helps the model understand inter-phase relationships.
    """

    def __init__(self, d_model, num_phases=3):
        """
        Args:
            d_model: Embedding dimension
            num_phases: Number of phases (3 for three-phase systems)
        """
        super(ThreePhaseEmbedding, self).__init__()

        self.d_model = d_model
        self.num_phases = num_phases

        # Phase angle embeddings (0, 120, 240 degrees)
        phase_angles = torch.tensor([0, 2*math.pi/3, 4*math.pi/3])
        self.register_buffer('phase_angles', phase_angles)

        # Learnable phase embedding
        self.phase_embed = nn.Embedding(num_phases, d_model)

        # Positive and negative sequence embeddings
        self.pos_seq_embed = nn.Linear(num_phases, d_model)
        self.neg_seq_embed = nn.Linear(num_phases, d_model)

    def forward(self, x, channel_ids=None):
        """
        Args:
            x: Input tensor [B, T, C] where C includes three-phase channels
            channel_ids: Optional tensor indicating which channels are Va, Vb, Vc

        Returns:
            Three-phase embedding [B, T, d_model]
        """
        B, T, C = x.size()

        # Assume first 3 channels are Va, Vb, Vc if not specified
        if channel_ids is None:
            voltage_channels = min(3, C)
        else:
            voltage_channels = len(channel_ids)

        # Get phase embeddings
        phase_ids = torch.arange(voltage_channels, device=x.device)
        phase_emb = self.phase_embed(phase_ids)  # [num_phases, d_model]

        # Calculate symmetrical components (simplified)
        # Positive sequence: Va + a*Vb + a²*Vc where a = exp(j*2π/3)
        if voltage_channels >= 3:
            v_abc = x[:, :, :3]  # [B, T, 3]

            # Positive sequence embedding
            pos_emb = self.pos_seq_embed(v_abc)  # [B, T, d_model]

            # Negative sequence: Va + a²*Vb + a*Vc
            v_acb = torch.stack([x[:, :, 0], x[:, :, 2], x[:, :, 1]], dim=-1)
            neg_emb = self.neg_seq_embed(v_acb)  # [B, T, d_model]

            # Combine with phase embeddings
            combined = pos_emb + 0.1 * neg_emb + phase_emb.mean(0).unsqueeze(0).unsqueeze(0)
        else:
            combined = phase_emb[:voltage_channels].mean(0).unsqueeze(0).unsqueeze(0)
            combined = combined.expand(B, T, -1)

        return combined


class VoltageQualityEmbedding(nn.Module):
    """
    Embedding that encodes voltage quality indicators.

    Key voltage quality metrics:
    - Voltage deviation from nominal
    - Total Harmonic Distortion (THD)
    - Voltage unbalance factor
    - Frequency deviation
    """

    def __init__(self, d_model, nominal_voltage=220.0, nominal_freq=50.0):
        """
        Args:
            d_model: Embedding dimension
            nominal_voltage: Nominal voltage value (V)
            nominal_freq: Nominal frequency (Hz)
        """
        super(VoltageQualityEmbedding, self).__init__()

        self.d_model = d_model
        self.nominal_voltage = nominal_voltage
        self.nominal_freq = nominal_freq

        # Quality indicator projections
        self.voltage_deviation_proj = nn.Linear(1, d_model // 4)
        self.thd_proj = nn.Linear(1, d_model // 4)
        self.unbalance_proj = nn.Linear(1, d_model // 4)
        self.freq_deviation_proj = nn.Linear(1, d_model // 4)

        # Combination projection
        self.combine_proj = nn.Linear(d_model, d_model)

    def forward(self, voltage, thd=None, unbalance=None, freq=None):
        """
        Args:
            voltage: Voltage values [B, T, num_phases]
            thd: Total harmonic distortion [B, T, 1] or None
            unbalance: Voltage unbalance factor [B, T, 1] or None
            freq: System frequency [B, T, 1] or None

        Returns:
            Voltage quality embedding [B, T, d_model]
        """
        B, T, _ = voltage.size()

        # Calculate voltage deviation
        mean_voltage = voltage.mean(dim=-1, keepdim=True)
        voltage_dev = (mean_voltage - self.nominal_voltage) / self.nominal_voltage
        volt_emb = self.voltage_deviation_proj(voltage_dev)

        # THD embedding
        if thd is not None:
            thd_emb = self.thd_proj(thd)
        else:
            thd_emb = torch.zeros(B, T, self.d_model // 4, device=voltage.device)

        # Unbalance embedding
        if unbalance is not None:
            unb_emb = self.unbalance_proj(unbalance)
        else:
            # Calculate simple unbalance from voltage
            if voltage.size(-1) >= 3:
                v_mean = voltage[:, :, :3].mean(dim=-1, keepdim=True)
                v_max_dev = (voltage[:, :, :3] - v_mean).abs().max(dim=-1, keepdim=True)[0]
                unb = v_max_dev / (v_mean + 1e-8) * 100
                unb_emb = self.unbalance_proj(unb)
            else:
                unb_emb = torch.zeros(B, T, self.d_model // 4, device=voltage.device)

        # Frequency deviation embedding
        if freq is not None:
            freq_dev = (freq - self.nominal_freq) / self.nominal_freq
            freq_emb = self.freq_deviation_proj(freq_dev)
        else:
            freq_emb = torch.zeros(B, T, self.d_model // 4, device=voltage.device)

        # Combine all quality embeddings
        combined = torch.cat([volt_emb, thd_emb, unb_emb, freq_emb], dim=-1)
        return self.combine_proj(combined)


class VoltageDataEmbedding(nn.Module):
    """
    Complete data embedding for voltage anomaly detection.

    Combines:
    1. Token embedding (from raw values)
    2. Power frequency embedding
    3. Daily load embedding
    4. Three-phase embedding
    5. Voltage quality embedding
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,
                 use_power_freq=True, use_daily=True, use_three_phase=True,
                 use_quality=True, sample_rate=1.0):
        """
        Args:
            c_in: Number of input channels
            d_model: Embedding dimension
            embed_type: Type of temporal embedding
            freq: Frequency of data
            dropout: Dropout rate
            use_power_freq: Whether to use power frequency embedding
            use_daily: Whether to use daily load embedding
            use_three_phase: Whether to use three-phase embedding
            use_quality: Whether to use voltage quality embedding
            sample_rate: Sampling rate in Hz
        """
        super(VoltageDataEmbedding, self).__init__()

        self.d_model = d_model
        self.use_power_freq = use_power_freq
        self.use_daily = use_daily
        self.use_three_phase = use_three_phase
        self.use_quality = use_quality

        # Token embedding (1D convolution)
        padding = 1
        self.token_conv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', bias=False
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode='fan_in', nonlinearity='leaky_relu')

        # Optional embeddings
        if use_power_freq:
            self.power_freq_embedding = PowerFrequencyEmbedding(d_model, sample_rate=sample_rate)

        if use_daily:
            # Samples per day depends on sampling rate
            samples_per_day = int(86400 * sample_rate)
            self.daily_embedding = DailyLoadEmbedding(d_model, samples_per_day=samples_per_day)

        if use_three_phase:
            self.three_phase_embedding = ThreePhaseEmbedding(d_model)

        if use_quality:
            self.quality_embedding = VoltageQualityEmbedding(d_model)

        # Combination weights
        num_embeddings = 1 + use_power_freq + use_daily + use_three_phase + use_quality
        self.combination_weights = nn.Parameter(torch.ones(num_embeddings) / num_embeddings)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        Args:
            x: Input tensor [B, T, C]
            x_mark: Optional temporal marks [B, T, mark_dim]

        Returns:
            Embedded tensor [B, T, d_model]
        """
        # Token embedding
        token_emb = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)

        embeddings = [token_emb]
        weights = [self.combination_weights[0]]

        idx = 1

        # Power frequency embedding
        if self.use_power_freq:
            pf_emb = self.power_freq_embedding(x)
            embeddings.append(pf_emb)
            weights.append(self.combination_weights[idx])
            idx += 1

        # Daily embedding
        if self.use_daily:
            daily_emb = self.daily_embedding(x)
            embeddings.append(daily_emb)
            weights.append(self.combination_weights[idx])
            idx += 1

        # Three-phase embedding
        if self.use_three_phase:
            tp_emb = self.three_phase_embedding(x)
            embeddings.append(tp_emb)
            weights.append(self.combination_weights[idx])
            idx += 1

        # Quality embedding
        if self.use_quality:
            # Extract voltage channels (assume first 3)
            voltage = x[:, :, :min(3, x.size(-1))]
            qual_emb = self.quality_embedding(voltage)
            embeddings.append(qual_emb)
            weights.append(self.combination_weights[idx])

        # Weighted combination
        weights = F.softmax(torch.stack(weights), dim=0)
        combined = sum(w * e for w, e in zip(weights, embeddings))

        return self.dropout(combined)
