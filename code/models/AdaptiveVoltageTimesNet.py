"""
AdaptiveVoltageTimesNet: 自适应序列长度的电压异常检测模型

主要改进:
1. 自适应周期发现: 根据seq_len动态调整预设周期，确保领域知识始终有效
2. 采样率感知: 支持不同采样率的数据（1Hz, 0.1Hz等）
3. 相对周期策略: 当绝对周期无效时，使用基于信号处理理论的相对周期
4. GPU优化: 减少GPU-CPU数据传输，提升计算效率

作者: 研究生论文项目
"""

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


def FFT_for_Period_Adaptive(x, k=2, preset_periods=None, preset_weight=0.3, device=None):
    """
    自适应周期发现函数 - GPU优化版本

    Args:
        x: 输入张量 [B, T, C]
        k: 选择的周期数量
        preset_periods: 预设周期列表（已根据seq_len过滤）
        preset_weight: 预设周期权重 (0-1)
        device: 计算设备

    Returns:
        period_list: 选定的周期列表 (numpy array)
        period_weight: 各周期权重 [B, k]
    """
    B, T, C = x.size()
    if device is None:
        device = x.device

    # FFT计算频谱
    xf = torch.fft.rfft(x, dim=1)

    # 计算各频率分量的平均幅值
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 移除直流分量

    # 获取top-k FFT发现的频率
    _, top_indices = torch.topk(frequency_list, min(k, len(frequency_list) - 1))

    # 在GPU上计算周期 (避免CPU传输)
    fft_periods = torch.clamp(T // (top_indices.float() + 1e-8), min=2, max=T // 2).int()

    # 组合预设周期和FFT周期
    if preset_periods is not None and len(preset_periods) > 0:
        n_preset = max(1, int(k * preset_weight))
        n_fft = k - n_preset

        # 预设周期张量
        preset_tensor = torch.tensor(preset_periods[:n_preset], device=device, dtype=torch.int)

        # 合并周期（去重）
        combined = torch.cat([fft_periods[:n_fft], preset_tensor])
        unique_periods = torch.unique(combined)

        # 确保有k个周期
        if len(unique_periods) < k:
            # 补充更多FFT周期
            additional = fft_periods[~torch.isin(fft_periods, unique_periods)]
            unique_periods = torch.cat([unique_periods, additional[:k - len(unique_periods)]])

        final_periods = unique_periods[:k]
    else:
        final_periods = fft_periods[:k]

    # 转换为numpy用于后续处理
    period_list = final_periods.cpu().numpy()

    # 计算周期权重（基于对应频率的幅值）
    freq_indices = torch.clamp(T // final_periods.float(), min=1, max=xf.shape[1] - 1).long()
    xf_amplitudes = abs(xf).mean(-1)  # [B, T//2+1]
    period_weight = xf_amplitudes[:, freq_indices]  # [B, k]

    return period_list, period_weight


class AdaptiveTimesBlock(nn.Module):
    """
    自适应TimesBlock - 根据序列长度动态调整周期策略
    """

    def __init__(self, configs, preset_periods=None, preset_weight=0.3):
        super(AdaptiveTimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.k = configs.top_k
        self.preset_weight = preset_weight
        self.preset_periods = preset_periods if preset_periods else []

        # 2D卷积用于周期特征提取
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

        # 时域平滑卷积（深度可分离卷积，减少参数）
        self.temporal_conv = nn.Conv1d(
            configs.d_model,
            configs.d_model,
            kernel_size=3,
            padding=1,
            groups=configs.d_model,
        )

    def forward(self, x):
        B, T, N = x.size()

        # 自适应周期发现
        period_list, period_weight = FFT_for_Period_Adaptive(
            x, self.k,
            preset_periods=self.preset_periods,
            preset_weight=self.preset_weight,
            device=x.device
        )

        res = []
        for i in range(len(period_list)):
            period = int(period_list[i])
            period = max(2, min(period, T // 2))  # 确保周期有效

            # 填充使序列长度可被周期整除
            total_len = self.seq_len + self.pred_len
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros([B, length - total_len, N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            # 1D -> 2D 重塑: [B, T, N] -> [B, N, T//period, period]
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()

            # 2D卷积
            out = self.conv(out)

            # 2D -> 1D 还原
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)

        # 自适应加权聚合 (使用expand而非repeat)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).expand(-1, T, N, -1)
        res = torch.sum(res * period_weight, dim=-1)

        # 时域平滑
        res = res + self.temporal_conv(res.permute(0, 2, 1)).permute(0, 2, 1)

        # 残差连接
        res = res + x

        return res


class Model(nn.Module):
    """
    AdaptiveVoltageTimesNet: 自适应电压异常检测模型

    针对农村低压配电网电压数据设计，支持:
    - 16维特征输入（三相电压电流、功率指标、电能质量指标）
    - 自适应序列长度（seq_len从20到10000均可有效工作）
    - 多种采样率（1Hz, 0.1Hz等）
    """

    # 电网预设周期（秒）
    GRID_PERIODS_SECONDS = [60, 300, 900, 3600]  # 1分钟, 5分钟, 15分钟, 1小时

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.pred_len = getattr(configs, 'pred_len', 0)

        # 采样率（默认1Hz，即每秒1个样本）
        self.sampling_rate = getattr(configs, 'sampling_rate', 1.0)

        # 预设周期权重（alpha参数，控制FFT vs 预设周期的比例）
        self.preset_weight = getattr(configs, 'preset_weight', 0.3)

        # 自适应计算预设周期
        self.preset_periods = self._calculate_adaptive_periods()

        # 编码器层
        self.model = nn.ModuleList([
            AdaptiveTimesBlock(configs, self.preset_periods, self.preset_weight)
            for _ in range(configs.e_layers)
        ])

        # 数据嵌入层
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 任务特定的输出头
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _calculate_adaptive_periods(self):
        """
        自适应计算有效的预设周期

        策略:
        1. 尝试使用绝对周期（基于采样率转换）
        2. 如果绝对周期都无效，使用基于信号处理理论的相对周期
        3. 相对周期选择: 5%, 10%, 20%, 33% 的序列长度
        """
        seq_len = self.seq_len
        max_valid_period = seq_len // 2  # 奈奎斯特准则

        periods = []

        # 尝试绝对周期（转换为样本数）
        for t_seconds in self.GRID_PERIODS_SECONDS:
            p = int(t_seconds * self.sampling_rate)
            if 2 <= p <= max_valid_period:
                periods.append(p)

        # 如果没有有效的绝对周期，使用相对周期
        if not periods:
            # 基于信号处理理论的比例选择
            # 这些比例对应常见的周期性模式
            ratios = [0.05, 0.1, 0.2, 0.33]
            for r in ratios:
                p = max(2, int(seq_len * r))
                if p <= max_valid_period and p not in periods:
                    periods.append(p)

        # 确保至少有2个周期
        if len(periods) < 2:
            periods = [max(2, seq_len // 10), max(2, seq_len // 4)]

        # 去重并排序
        periods = sorted(list(set(periods)))

        return periods

    def anomaly_detection(self, x_enc):
        """异常检测任务 - 学习正常模式，重构误差大则为异常"""
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # 嵌入
        enc_out = self.enc_embedding(x_enc, None)

        # 编码器层
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 投影
        dec_out = self.projection(enc_out)

        # 反归一化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.seq_len + self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.seq_len + self.pred_len, -1)

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """预测任务"""
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len + self.seq_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len + self.seq_len, -1)

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """缺失值填补任务"""
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

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len + self.seq_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len + self.seq_len, -1)

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """分类任务"""
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
        """前向传播 - 根据任务类型路由"""
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
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
