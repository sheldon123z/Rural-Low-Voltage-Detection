"""
ThreePhaseAttention: Attention Mechanisms for Three-Phase Power Systems

This module provides attention mechanisms specifically designed for analyzing
three-phase voltage signals in power grid anomaly detection:

1. InterPhaseAttention: Captures relationships between Va, Vb, Vc phases
2. SymmetricalComponentAttention: Analyzes positive/negative/zero sequences
3. TransientAttention: Multi-scale attention for transient event detection
4. VoltageChannelAttention: Channel-wise attention for voltage features

Key concepts:
- In balanced systems, Va, Vb, Vc have 120° phase difference
- Unbalance indicates issues (asymmetric loads, faults)
- Symmetrical components help diagnose fault types
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterPhaseAttention(nn.Module):
    """
    Attention mechanism that captures inter-phase relationships.

    In three-phase systems:
    - Normal: Va, Vb, Vc are balanced with 120° phase shifts
    - Fault: Phase relationships deviate from normal patterns

    This attention helps detect anomalies by modeling phase interactions.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1, num_phases=3):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            num_phases: Number of phases (default 3 for three-phase)
        """
        super(InterPhaseAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_phases = num_phases
        self.d_k = d_model // n_heads

        # Query, Key, Value projections for each phase
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Phase-specific transformations
        self.phase_transforms = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_phases)]
        )

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Phase relationship encoding (learnable)
        # Initialize with 120° phase shifts
        phase_angles = torch.tensor([0, 2 * math.pi / 3, 4 * math.pi / 3])
        self.register_buffer("phase_angles", phase_angles)
        self.phase_bias = nn.Parameter(torch.zeros(num_phases, num_phases))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, phase_mask=None):
        """
        Args:
            x: Input tensor [B, T, C] where C = num_phases * features_per_phase
            phase_mask: Optional mask for phase interactions [num_phases, num_phases]

        Returns:
            Attention output [B, T, d_model]
        """
        B, T, C = x.size()

        # Project inputs
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k)

        # Transpose for attention: [B, n_heads, T, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add phase relationship bias
        # This encourages the model to learn phase-specific interactions
        if self.num_phases <= T:
            phase_bias_expanded = self.phase_bias.unsqueeze(0).unsqueeze(0)
            # Tile to match temporal dimension
            n_tiles = (T + self.num_phases - 1) // self.num_phases
            phase_bias_tiled = phase_bias_expanded.repeat(1, 1, n_tiles, n_tiles)
            phase_bias_tiled = phase_bias_tiled[:, :, :T, :T]
            scores = scores + phase_bias_tiled

        if phase_mask is not None:
            scores = scores.masked_fill(phase_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_o(context)

        # Residual connection and layer norm
        output = self.layer_norm(output + x)

        return output


class SymmetricalComponentAttention(nn.Module):
    """
    Attention based on symmetrical component analysis.

    Symmetrical components decompose unbalanced three-phase systems into:
    - Positive sequence (balanced, normal operation)
    - Negative sequence (indicates unbalance)
    - Zero sequence (indicates ground faults)

    This attention helps identify different types of power system faults.
    """

    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super(SymmetricalComponentAttention, self).__init__()

        self.d_model = d_model

        # Fortescue transformation matrix for symmetrical components
        # a = exp(j*2π/3) = -0.5 + j*0.866
        a_real = -0.5
        a_imag = math.sqrt(3) / 2

        # Transformation matrix (real and imaginary parts)
        # [1, 1, 1; 1, a², a; 1, a, a²] for positive, negative, zero
        self.register_buffer(
            "fortescue_real",
            torch.tensor(
                [
                    [1, 1, 1],
                    [1, a_real**2 - a_imag**2, a_real],
                    [1, a_real, a_real**2 - a_imag**2],
                ]
            )
            / 3.0,
        )

        self.register_buffer(
            "fortescue_imag",
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 2 * a_real * a_imag, a_imag],
                    [0, a_imag, 2 * a_real * a_imag],
                ]
            )
            / 3.0,
        )

        # Sequence-specific attention
        self.pos_seq_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.neg_seq_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.zero_seq_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )

        # Sequence weighting (learnable importance of each sequence)
        self.sequence_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3]))

        # Output projection
        self.output_proj = nn.Linear(d_model * 3, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def compute_symmetrical_components(self, x):
        """
        Compute symmetrical components from three-phase signals.

        Args:
            x: Three-phase signals [B, T, 3]

        Returns:
            Tuple of (positive, negative, zero) sequences, each [B, T, 1]
        """
        B, T, _ = x.size()

        # Apply Fortescue transformation (simplified real-valued version)
        # For real signals, we approximate the transformation
        x_transformed = torch.matmul(x, self.fortescue_real.T)

        pos_seq = x_transformed[:, :, 0:1]  # Positive sequence
        neg_seq = x_transformed[:, :, 1:2]  # Negative sequence
        zero_seq = x_transformed[:, :, 2:3]  # Zero sequence

        return pos_seq, neg_seq, zero_seq

    def forward(self, x, return_sequences=False):
        """
        Args:
            x: Input tensor [B, T, d_model]
            return_sequences: Whether to return individual sequence outputs

        Returns:
            Attention output [B, T, d_model]
            Optionally: (pos_out, neg_out, zero_out) if return_sequences=True
        """
        B, T, _ = x.size()

        # Apply attention for each sequence type
        pos_out, _ = self.pos_seq_attn(x, x, x)
        neg_out, _ = self.neg_seq_attn(x, x, x)
        zero_out, _ = self.zero_seq_attn(x, x, x)

        # Weighted combination based on sequence importance
        weights = F.softmax(self.sequence_weights, dim=0)
        combined = torch.cat(
            [weights[0] * pos_out, weights[1] * neg_out, weights[2] * zero_out], dim=-1
        )

        # Project back to d_model
        output = self.output_proj(combined)
        output = self.dropout(output)

        # Residual connection and layer norm
        output = self.layer_norm(output + x)

        if return_sequences:
            return output, (pos_out, neg_out, zero_out)
        return output


class TransientAttention(nn.Module):
    """
    Multi-scale attention for detecting transient events in voltage signals.

    Transient events in power systems include:
    - Voltage sags (ms to seconds)
    - Voltage swells
    - Momentary interruptions
    - Switching transients (μs to ms)

    This attention uses multiple time scales to capture different transient types.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1, scales=[1, 3, 5, 10]):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            scales: List of temporal scales for multi-scale attention
        """
        super(TransientAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales

        # Multi-scale convolutions for different transient durations
        self.scale_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model, d_model, kernel_size=s, padding=s // 2, groups=d_model
                )
                for s in scales
            ]
        )

        # Scale-specific attention
        self.scale_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model, num_heads=n_heads, dropout=dropout, batch_first=True
                )
                for _ in scales
            ]
        )

        # Scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

        # Output projection
        self.output_proj = nn.Linear(d_model * len(scales), d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, d_model]

        Returns:
            Multi-scale attention output [B, T, d_model]
        """
        B, T, D = x.size()

        scale_outputs = []

        for i, (conv, attn) in enumerate(zip(self.scale_convs, self.scale_attentions)):
            # Apply scale-specific convolution
            x_conv = conv(x.permute(0, 2, 1)).permute(0, 2, 1)

            # Ensure same length
            if x_conv.size(1) != T:
                x_conv = x_conv[:, :T, :]

            # Apply attention at this scale
            scale_out, _ = attn(x_conv, x_conv, x_conv)
            scale_outputs.append(scale_out)

        # Weighted combination of scales
        weights = F.softmax(self.scale_weights, dim=0)
        combined = torch.cat(
            [w * out for w, out in zip(weights, scale_outputs)], dim=-1
        )

        # Project to output dimension
        output = self.output_proj(combined)
        output = self.dropout(output)

        # Residual connection and layer norm
        output = self.layer_norm(output + x)

        return output


class VoltageChannelAttention(nn.Module):
    """
    Channel-wise attention for voltage feature selection.

    Different voltage features have varying importance for anomaly detection:
    - Va, Vb, Vc: Direct voltage measurements
    - Ia, Ib, Ic: Current measurements
    - P, Q, S: Power metrics
    - THD: Harmonic distortion
    - Unbalance: Phase unbalance factor

    This attention learns to weight different features based on their
    relevance to anomaly detection.
    """

    def __init__(self, num_channels, reduction_ratio=4):
        """
        Args:
            num_channels: Number of input channels/features
            reduction_ratio: Reduction ratio for the bottleneck
        """
        super(VoltageChannelAttention, self).__init__()

        self.num_channels = num_channels
        reduced_channels = max(1, num_channels // reduction_ratio)

        # Channel attention with squeeze-and-excitation style
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, num_channels, bias=False),
        )

        # Feature group weighting (voltage, current, power, quality)
        # Learnable importance for different feature groups
        self.group_weights = nn.Parameter(torch.ones(4))

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]

        Returns:
            Channel-attended output [B, T, C]
        """
        B, T, C = x.size()

        # Global average and max pooling
        x_permuted = x.permute(0, 2, 1)  # [B, C, T]
        avg_out = self.avg_pool(x_permuted).squeeze(-1)  # [B, C]
        max_out = self.max_pool(x_permuted).squeeze(-1)  # [B, C]

        # Channel attention weights
        avg_attn = self.fc(avg_out)
        max_attn = self.fc(max_out)
        attn = torch.sigmoid(avg_attn + max_attn)  # [B, C]

        # Apply attention
        output = x * attn.unsqueeze(1)

        return output


class VoltageAttentionBlock(nn.Module):
    """
    Combined attention block for voltage anomaly detection.

    Integrates multiple attention mechanisms:
    1. Inter-phase attention for phase relationships
    2. Transient attention for multi-scale temporal patterns
    3. Channel attention for feature importance

    This provides comprehensive attention for power grid signals.
    """

    def __init__(
        self,
        d_model,
        num_channels,
        n_heads=4,
        dropout=0.1,
        use_inter_phase=True,
        use_transient=True,
        use_channel=True,
    ):
        """
        Args:
            d_model: Model dimension
            num_channels: Number of input channels
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_inter_phase: Whether to use inter-phase attention
            use_transient: Whether to use transient attention
            use_channel: Whether to use channel attention
        """
        super(VoltageAttentionBlock, self).__init__()

        self.use_inter_phase = use_inter_phase
        self.use_transient = use_transient
        self.use_channel = use_channel

        if use_inter_phase:
            self.inter_phase_attn = InterPhaseAttention(d_model, n_heads, dropout)

        if use_transient:
            self.transient_attn = TransientAttention(d_model, n_heads, dropout)

        if use_channel:
            self.channel_attn = VoltageChannelAttention(num_channels)

        # Final projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, x_raw=None):
        """
        Args:
            x: Embedded input [B, T, d_model]
            x_raw: Raw input for channel attention [B, T, num_channels] (optional)

        Returns:
            Attention output [B, T, d_model]
        """
        output = x

        # Apply inter-phase attention
        if self.use_inter_phase:
            output = self.inter_phase_attn(output)

        # Apply transient attention
        if self.use_transient:
            output = self.transient_attn(output)

        # Apply channel attention on raw features if provided
        if self.use_channel and x_raw is not None:
            channel_weights = self.channel_attn(x_raw)
            # Broadcast channel weights to embedded dimension
            # This is a simplified application; in practice, might need adaptation
            output = output * channel_weights.mean(dim=-1, keepdim=True).expand_as(
                output
            )

        # Final projection with residual
        output = self.output_proj(output)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output
