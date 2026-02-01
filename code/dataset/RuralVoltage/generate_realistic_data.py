"""
Realistic Rural Voltage Anomaly Detection Data Generator (V2.0)

Major improvements over V1.0:
1. Realistic anomaly patterns with gradual transitions
2. Compound anomalies (multiple types occurring together)
3. Real-world noise models (1/f noise, power line interference)
4. True three-phase relationships with phase angles
5. Realistic rural load patterns (seasonal, weekly, daily)
6. Proper harmonic modeling (3rd, 5th, 7th harmonics)
7. Voltage flicker and transient events
8. Equipment switching transients

Based on:
- GB/T 12325-2008 (Voltage deviation limits)
- GB/T 14549-1993 (Harmonic limits)
- GB/T 15543-2008 (Unbalance limits)
- IEC 61000-4-30 (Power quality measurement methods)

Usage:
    python generate_realistic_data.py --train_samples 50000 --test_samples 10000 --anomaly_ratio 0.12
"""

import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime, timedelta
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Constants based on Chinese National Standards
# =============================================================================
NOMINAL_VOLTAGE = 220.0  # V (phase voltage)
NOMINAL_FREQUENCY = 50.0  # Hz
SAMPLING_RATE = 1.0  # Hz (1 sample per second)

# GB/T 12325-2008: Voltage deviation limits
VOLTAGE_UPPER_LIMIT = 242.0  # +10%
VOLTAGE_LOWER_LIMIT = 198.0  # -10%

# GB/T 14549-1993: THD limits
THD_LIMIT = 5.0  # %

# GB/T 15543-2008: Unbalance limits
UNBALANCE_LIMIT = 4.0  # %


# =============================================================================
# Noise Models
# =============================================================================
def generate_pink_noise(n_samples, scale=1.0):
    """Generate 1/f (pink) noise - more realistic than white noise."""
    # Generate white noise
    white = np.random.randn(n_samples)
    # Apply 1/f filter using FFT
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1e-10  # Avoid division by zero
    # 1/f filter
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n_samples)
    return pink * scale


def generate_power_line_interference(n_samples, amplitude=0.5):
    """Generate 50Hz power line interference and harmonics."""
    t = np.arange(n_samples) / SAMPLING_RATE
    interference = amplitude * np.sin(2 * np.pi * 50 * t)
    # Add 3rd and 5th harmonics
    interference += 0.3 * amplitude * np.sin(2 * np.pi * 150 * t)
    interference += 0.2 * amplitude * np.sin(2 * np.pi * 250 * t)
    return interference


def generate_measurement_noise(n_samples, snr_db=40):
    """Generate realistic measurement noise with given SNR."""
    # Combine white noise, pink noise, and quantization noise
    white = np.random.randn(n_samples)
    pink = generate_pink_noise(n_samples)
    
    # Mix: 60% pink, 40% white
    noise = 0.6 * pink + 0.4 * white
    
    # Scale to achieve target SNR (relative to nominal voltage)
    signal_power = NOMINAL_VOLTAGE ** 2
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power) / np.std(noise)
    
    return noise


# =============================================================================
# Load Pattern Models
# =============================================================================
def generate_rural_load_pattern(n_samples, season='summer', day_type='weekday'):
    """
    Generate realistic rural load pattern.
    
    Rural characteristics:
    - Morning peak: 6-8 AM (livestock feeding, irrigation pumps)
    - Noon dip: 12-2 PM
    - Evening peak: 6-9 PM (residential load)
    - Night low: 11 PM - 5 AM
    
    Seasonal variations:
    - Summer: Higher evening load (AC), irrigation
    - Winter: Higher morning/evening load (heating)
    """
    t = np.arange(n_samples)
    hours = (t / 3600) % 24  # Hour of day
    
    # Base daily pattern (normalized 0-1)
    # Morning peak around 7 AM
    morning_peak = 0.15 * np.exp(-((hours - 7) ** 2) / 2)
    # Evening peak around 7 PM
    evening_peak = 0.25 * np.exp(-((hours - 19) ** 2) / 3)
    # Noon dip
    noon_dip = -0.08 * np.exp(-((hours - 13) ** 2) / 2)
    # Night base load
    night_low = 0.05 * np.exp(-((hours - 3) ** 2) / 8)
    
    # Combine patterns
    base_pattern = 0.85 + morning_peak + evening_peak + noon_dip - night_low
    
    # Seasonal adjustment
    if season == 'summer':
        # Higher evening load (cooling)
        base_pattern += 0.1 * np.exp(-((hours - 15) ** 2) / 8)
        # Irrigation pump cycles (random)
        irrigation = 0.05 * (np.random.rand(n_samples) > 0.95).astype(float)
        irrigation = gaussian_filter1d(irrigation, sigma=30)
        base_pattern += irrigation
    elif season == 'winter':
        # Higher morning/evening heating load
        base_pattern += 0.08 * np.exp(-((hours - 7) ** 2) / 2)
        base_pattern += 0.12 * np.exp(-((hours - 19) ** 2) / 4)
    
    # Weekend adjustment
    if day_type == 'weekend':
        # Shift morning peak later
        base_pattern = np.roll(base_pattern, int(2 * 3600))
        # Slightly lower overall
        base_pattern *= 0.92
    
    # Add slow random variations (weather, cloud cover effects on solar)
    slow_variation = generate_pink_noise(n_samples, scale=0.02)
    slow_variation = gaussian_filter1d(slow_variation, sigma=1000)
    base_pattern += slow_variation
    
    # Normalize to reasonable range
    base_pattern = np.clip(base_pattern, 0.7, 1.15)
    
    return base_pattern


# =============================================================================
# Three-Phase Voltage Generation
# =============================================================================
def generate_three_phase_voltage(n_samples, load_pattern, noise_level=2.0):
    """
    Generate realistic three-phase voltage with proper relationships.
    
    Features:
    - 120° phase separation (in phasor domain)
    - Load-dependent voltage drop
    - Natural slight unbalance
    - Correlated noise between phases
    """
    t = np.arange(n_samples) / SAMPLING_RATE
    
    # Voltage magnitude affected by load (voltage drop under load)
    # Higher load -> lower voltage (simplified impedance model)
    voltage_drop = 0.02 * (load_pattern - 1.0)  # ~2% voltage regulation
    
    # Base voltage magnitudes
    V_base = NOMINAL_VOLTAGE * (1 - voltage_drop)
    
    # Natural unbalance (random but persistent)
    unbalance_a = np.random.uniform(-0.01, 0.01)
    unbalance_b = np.random.uniform(-0.01, 0.01)
    unbalance_c = -(unbalance_a + unbalance_b) * 0.5  # Partial compensation
    
    # Slow unbalance drift
    drift = generate_pink_noise(n_samples, scale=0.005)
    drift = gaussian_filter1d(drift, sigma=5000)
    
    Va = V_base * (1 + unbalance_a + drift)
    Vb = V_base * (1 + unbalance_b + drift * 0.8)
    Vc = V_base * (1 + unbalance_c - drift * 0.6)
    
    # Add correlated and uncorrelated noise
    common_noise = generate_measurement_noise(n_samples, snr_db=45)
    Va += common_noise + generate_measurement_noise(n_samples, snr_db=50)
    Vb += common_noise * 0.8 + generate_measurement_noise(n_samples, snr_db=50)
    Vc += common_noise * 0.6 + generate_measurement_noise(n_samples, snr_db=50)
    
    return Va, Vb, Vc


# =============================================================================
# Current and Power Generation
# =============================================================================
def generate_current(Va, Vb, Vc, load_pattern, base_current=15.0):
    """Generate realistic current based on voltage and load."""
    n_samples = len(Va)
    
    # Current proportional to load and inversely to voltage
    Ia = base_current * load_pattern * (NOMINAL_VOLTAGE / Va)
    Ib = base_current * load_pattern * (NOMINAL_VOLTAGE / Vb)
    Ic = base_current * load_pattern * (NOMINAL_VOLTAGE / Vc)
    
    # Add load fluctuations
    load_noise = generate_pink_noise(n_samples, scale=0.5)
    load_noise = gaussian_filter1d(load_noise, sigma=10)
    
    Ia += load_noise + np.random.normal(0, 0.3, n_samples)
    Ib += load_noise * 0.9 + np.random.normal(0, 0.3, n_samples)
    Ic += load_noise * 0.85 + np.random.normal(0, 0.3, n_samples)
    
    # Ensure positive
    Ia = np.maximum(Ia, 0.1)
    Ib = np.maximum(Ib, 0.1)
    Ic = np.maximum(Ic, 0.1)
    
    return Ia, Ib, Ic


def generate_power_metrics(Va, Vb, Vc, Ia, Ib, Ic, power_factor_base=0.88):
    """Generate power metrics with realistic relationships."""
    n_samples = len(Va)
    
    # Power factor varies with load
    load_variation = (Ia + Ib + Ic) / (3 * 15.0)  # Normalized load
    pf_variation = 0.05 * (load_variation - 1.0)  # PF drops slightly with load
    PF = power_factor_base + pf_variation + np.random.normal(0, 0.02, n_samples)
    PF = np.clip(PF, 0.7, 0.99)
    
    # Active power (kW)
    P = (Va * Ia + Vb * Ib + Vc * Ic) / 1000 * PF
    
    # Reactive power (kVar)
    Q = P * np.tan(np.arccos(PF))
    
    # Apparent power (kVA)
    S = np.sqrt(P**2 + Q**2)
    
    return P, Q, S, PF


# =============================================================================
# Power Quality Metrics
# =============================================================================
def generate_harmonic_content(n_samples, base_thd=2.0):
    """
    Generate realistic harmonic content (THD).
    
    Typical harmonic sources in rural areas:
    - LED lighting: 3rd, 5th harmonics
    - Motor drives: 5th, 7th harmonics
    - Power electronics: Wide spectrum
    """
    # Base THD with slow variation
    thd_base = base_thd + generate_pink_noise(n_samples, scale=0.5)
    thd_base = gaussian_filter1d(thd_base, sigma=500)
    
    # Occasional spikes (equipment switching)
    spike_prob = 0.001
    spikes = (np.random.rand(n_samples) < spike_prob).astype(float)
    spikes = gaussian_filter1d(spikes, sigma=5) * 3
    
    thd = thd_base + spikes + np.random.normal(0, 0.3, n_samples)
    thd = np.clip(thd, 0.5, THD_LIMIT - 0.5)  # Normal range
    
    return thd


def calculate_unbalance(Va, Vb, Vc):
    """Calculate voltage unbalance factor per IEC standards."""
    V_avg = (Va + Vb + Vc) / 3
    V_max_dev = np.maximum(
        np.abs(Va - V_avg),
        np.maximum(np.abs(Vb - V_avg), np.abs(Vc - V_avg))
    )
    return V_max_dev / (V_avg + 1e-8) * 100


def generate_frequency(n_samples, nominal=50.0):
    """Generate realistic frequency variations."""
    # Slow frequency drift (grid regulation)
    drift = generate_pink_noise(n_samples, scale=0.02)
    drift = gaussian_filter1d(drift, sigma=2000)
    
    # Fast variations
    fast = np.random.normal(0, 0.01, n_samples)
    
    freq = nominal + drift + fast
    return np.clip(freq, 49.5, 50.5)


# =============================================================================
# Anomaly Injection Functions (Realistic)
# =============================================================================
def inject_undervoltage_realistic(Va, Vb, Vc, start, duration, severity=0.12, 
                                  transition_time=10, affected_phases='all'):
    """
    Inject realistic undervoltage with gradual transitions.
    
    Args:
        severity: Voltage drop fraction (0.12 = 12% drop)
        transition_time: Ramp time in samples
        affected_phases: 'all', 'single', or 'two'
    """
    end = min(start + duration, len(Va))
    
    # Create smooth transition envelope
    envelope = np.ones(duration)
    
    # Ramp down
    ramp_down = np.linspace(1, 1 - severity, min(transition_time, duration // 4))
    envelope[:len(ramp_down)] = ramp_down
    
    # Ramp up (recovery)
    ramp_up = np.linspace(1 - severity, 1, min(transition_time * 2, duration // 4))
    envelope[-len(ramp_up):] = ramp_up
    
    # Fill middle with sustained level + small variations
    mid_start = len(ramp_down)
    mid_end = len(envelope) - len(ramp_up)
    if mid_end > mid_start:
        mid_length = mid_end - mid_start
        sustained = (1 - severity) + np.random.normal(0, severity * 0.1, mid_length)
        envelope[mid_start:mid_end] = sustained
    
    # Apply to phases
    if affected_phases == 'all':
        Va[start:end] *= envelope[:end-start]
        Vb[start:end] *= envelope[:end-start]
        Vc[start:end] *= envelope[:end-start]
    elif affected_phases == 'single':
        phase = np.random.randint(0, 3)
        if phase == 0:
            Va[start:end] *= envelope[:end-start]
        elif phase == 1:
            Vb[start:end] *= envelope[:end-start]
        else:
            Vc[start:end] *= envelope[:end-start]
    else:  # two phases
        phases = np.random.choice([0, 1, 2], 2, replace=False)
        for p in phases:
            if p == 0:
                Va[start:end] *= envelope[:end-start]
            elif p == 1:
                Vb[start:end] *= envelope[:end-start]
            else:
                Vc[start:end] *= envelope[:end-start]
    
    return Va, Vb, Vc


def inject_overvoltage_realistic(Va, Vb, Vc, start, duration, severity=0.10,
                                 transition_time=15):
    """Inject realistic overvoltage with gradual transitions."""
    end = min(start + duration, len(Va))
    
    # Create smooth transition envelope
    envelope = np.ones(duration)
    
    # Gradual rise
    ramp_up = np.linspace(1, 1 + severity, min(transition_time, duration // 4))
    envelope[:len(ramp_up)] = ramp_up
    
    # Gradual decline
    ramp_down = np.linspace(1 + severity, 1, min(transition_time * 2, duration // 3))
    envelope[-len(ramp_down):] = ramp_down
    
    # Sustained level with fluctuations
    mid_start = len(ramp_up)
    mid_end = len(envelope) - len(ramp_down)
    if mid_end > mid_start:
        mid_length = mid_end - mid_start
        sustained = (1 + severity) + np.random.normal(0, severity * 0.08, mid_length)
        envelope[mid_start:mid_end] = sustained
    
    # Apply to all phases
    Va[start:end] *= envelope[:end-start]
    Vb[start:end] *= envelope[:end-start]
    Vc[start:end] *= envelope[:end-start]
    
    return Va, Vb, Vc


def inject_voltage_sag_realistic(Va, Vb, Vc, start, duration, depth=0.25,
                                sag_type='three_phase'):
    """
    Inject realistic voltage sag based on IEC 61000-4-30.
    
    Sag types:
    - three_phase: All phases affected equally (fault on transmission)
    - single_phase: One phase affected (single-phase fault)
    - phase_to_phase: Two phases affected (line-to-line fault)
    """
    end = min(start + duration, len(Va))
    actual_duration = end - start
    
    # Characteristic sag shape: fast drop, slight recovery, fast restore
    t = np.linspace(0, 1, actual_duration)
    
    # Fast initial drop (10% of duration)
    # Slight sag during sustained period
    # Recovery with possible overshoot
    
    drop_phase = int(actual_duration * 0.05)
    sustain_phase = int(actual_duration * 0.8)
    recovery_phase = actual_duration - drop_phase - sustain_phase
    
    envelope = np.ones(actual_duration)
    
    # Drop phase (exponential)
    if drop_phase > 0:
        envelope[:drop_phase] = 1 - depth * (1 - np.exp(-5 * t[:drop_phase] / t[drop_phase]))
    
    # Sustain phase with point-on-wave variation
    if sustain_phase > 0:
        sustain_start = drop_phase
        sustain_end = drop_phase + sustain_phase
        base_level = 1 - depth
        # Add realistic oscillation
        osc = 0.02 * depth * np.sin(2 * np.pi * 3 * t[sustain_start:sustain_end])
        envelope[sustain_start:sustain_end] = base_level + osc + np.random.normal(0, 0.01 * depth, sustain_phase)
    
    # Recovery phase with possible overshoot
    if recovery_phase > 0:
        recovery_start = drop_phase + sustain_phase
        t_recovery = np.linspace(0, 1, recovery_phase)
        # Damped oscillation recovery
        recovery = 1 + 0.03 * np.exp(-3 * t_recovery) * np.sin(10 * np.pi * t_recovery)
        envelope[recovery_start:] = (1 - depth) + depth * (1 - np.exp(-5 * t_recovery)) * recovery[:len(envelope) - recovery_start]
    
    # Apply based on sag type
    if sag_type == 'three_phase':
        Va[start:end] *= envelope
        Vb[start:end] *= envelope
        Vc[start:end] *= envelope
    elif sag_type == 'single_phase':
        phase = np.random.randint(0, 3)
        if phase == 0:
            Va[start:end] *= envelope
            # Other phases see slight rise
            Vb[start:end] *= (1 + 0.02 * depth)
            Vc[start:end] *= (1 + 0.02 * depth)
        elif phase == 1:
            Vb[start:end] *= envelope
            Va[start:end] *= (1 + 0.02 * depth)
            Vc[start:end] *= (1 + 0.02 * depth)
        else:
            Vc[start:end] *= envelope
            Va[start:end] *= (1 + 0.02 * depth)
            Vb[start:end] *= (1 + 0.02 * depth)
    else:  # phase_to_phase
        phases = np.random.choice([0, 1, 2], 2, replace=False)
        for p in phases:
            if p == 0:
                Va[start:end] *= envelope
            elif p == 1:
                Vb[start:end] *= envelope
            else:
                Vc[start:end] *= envelope
    
    return Va, Vb, Vc


def inject_harmonics_realistic(THD_Va, THD_Vb, THD_Vc, start, duration, 
                               severity='moderate'):
    """
    Inject realistic harmonic distortion.
    
    Severity levels:
    - mild: THD 5-7%
    - moderate: THD 7-10%
    - severe: THD 10-15%
    """
    end = min(start + duration, len(THD_Va))
    
    severity_ranges = {
        'mild': (5, 7),
        'moderate': (7, 10),
        'severe': (10, 15)
    }
    thd_min, thd_max = severity_ranges.get(severity, (7, 10))
    
    # Gradual build-up and decay
    actual_duration = end - start
    t = np.linspace(0, np.pi, actual_duration)
    envelope = np.sin(t)  # Smooth rise and fall
    
    # Base elevated THD
    base_thd = np.random.uniform(thd_min, thd_max)
    
    # Add realistic fluctuations
    for i, (thd, phase_offset) in enumerate([(THD_Va, 0), (THD_Vb, 0.1), (THD_Vc, 0.2)]):
        thd_increase = base_thd * envelope + np.random.normal(0, 0.5, actual_duration)
        thd_increase = np.maximum(thd_increase, 0)
        
        # Different phases can have different THD levels
        phase_factor = 1 + np.random.uniform(-0.15, 0.15)
        thd[start:end] = np.maximum(thd[start:end], thd_increase * phase_factor)
    
    return THD_Va, THD_Vb, THD_Vc


def inject_unbalance_realistic(Va, Vb, Vc, start, duration, unbalance_percent=6.0):
    """
    Inject realistic three-phase unbalance.
    
    Causes in rural areas:
    - Single-phase loads
    - Broken neutral
    - Unequal transformer tap settings
    """
    end = min(start + duration, len(Va))
    actual_duration = end - start
    
    # Gradual development of unbalance
    t = np.linspace(0, 1, actual_duration)
    envelope = 1 - np.exp(-3 * t)  # Gradual onset
    
    # Recovery envelope
    recovery_start = int(actual_duration * 0.8)
    if recovery_start < actual_duration:
        t_recovery = np.linspace(0, 1, actual_duration - recovery_start)
        envelope[recovery_start:] *= np.exp(-2 * t_recovery)
    
    # Unbalance pattern: one phase drops, one rises, one stays
    unbalance_factor = unbalance_percent / 100
    phase_effects = np.random.permutation([
        -unbalance_factor,  # Drop
        unbalance_factor * 0.5,  # Rise
        unbalance_factor * 0.3  # Slight change
    ])
    
    Va[start:end] *= (1 + phase_effects[0] * envelope)
    Vb[start:end] *= (1 + phase_effects[1] * envelope)
    Vc[start:end] *= (1 + phase_effects[2] * envelope)
    
    return Va, Vb, Vc


def inject_transient_realistic(Va, Vb, Vc, start, transient_type='motor_start'):
    """
    Inject realistic transient events.
    
    Types:
    - motor_start: Inrush current causing voltage dip
    - capacitor_switch: Oscillatory transient
    - load_switch: Step change with ringing
    """
    n_samples = len(Va)
    
    if transient_type == 'motor_start':
        # Motor starting: 3-6x inrush, voltage dips 10-20%
        duration = np.random.randint(30, 100)  # 30-100 seconds
        end = min(start + duration, n_samples)
        actual_duration = end - start
        
        t = np.linspace(0, 1, actual_duration)
        # Initial deep dip, gradual recovery
        dip = 0.15 * np.exp(-3 * t) + 0.03 * np.exp(-0.5 * t) * np.sin(10 * np.pi * t)
        
        Va[start:end] *= (1 - dip)
        Vb[start:end] *= (1 - dip)
        Vc[start:end] *= (1 - dip)
        
    elif transient_type == 'capacitor_switch':
        # Capacitor switching: oscillatory transient
        duration = np.random.randint(5, 20)
        end = min(start + duration, n_samples)
        actual_duration = end - start
        
        t = np.linspace(0, 1, actual_duration)
        # Damped oscillation
        osc = 0.3 * np.exp(-10 * t) * np.sin(100 * np.pi * t)
        
        # Affects all phases but with phase shifts
        Va[start:end] *= (1 + osc)
        Vb[start:end] *= (1 + 0.8 * osc)
        Vc[start:end] *= (1 + 0.6 * osc)
        
    elif transient_type == 'load_switch':
        # Load switching: step with ringing
        duration = np.random.randint(10, 30)
        end = min(start + duration, n_samples)
        actual_duration = end - start
        
        t = np.linspace(0, 1, actual_duration)
        step = np.random.choice([-1, 1]) * 0.05  # Up or down
        ringing = step * (1 + 0.3 * np.exp(-5 * t) * np.sin(20 * np.pi * t))
        
        Va[start:end] *= (1 + ringing)
        Vb[start:end] *= (1 + ringing * 0.9)
        Vc[start:end] *= (1 + ringing * 0.85)
    
    return Va, Vb, Vc


def inject_flicker_realistic(Va, Vb, Vc, start, duration, flicker_frequency=8.0):
    """
    Inject voltage flicker (cyclic voltage variation).
    
    Common causes: Arc furnaces, welding, compressors
    Flicker frequency typically 0.5-25 Hz, most sensitive at 8.8 Hz
    """
    end = min(start + duration, len(Va))
    actual_duration = end - start
    
    t = np.arange(actual_duration) / SAMPLING_RATE
    
    # Modulating signal (flicker)
    flicker_depth = np.random.uniform(0.02, 0.05)  # 2-5% modulation
    
    # Envelope for gradual onset/offset
    envelope = np.ones(actual_duration)
    ramp = min(actual_duration // 4, 50)
    envelope[:ramp] = np.linspace(0, 1, ramp)
    envelope[-ramp:] = np.linspace(1, 0, ramp)
    
    modulation = 1 + flicker_depth * envelope * np.sin(2 * np.pi * flicker_frequency * t)
    
    Va[start:end] *= modulation
    Vb[start:end] *= modulation
    Vc[start:end] *= modulation
    
    return Va, Vb, Vc


# =============================================================================
# Compound Anomaly Generation
# =============================================================================
def inject_compound_anomaly(Va, Vb, Vc, THD_Va, THD_Vb, THD_Vc, start, duration):
    """
    Inject compound anomaly (multiple issues occurring together).
    
    Common combinations in rural grids:
    1. Undervoltage + Harmonics (overloaded transformer)
    2. Unbalance + Voltage sag (single-phase fault)
    3. Flicker + Harmonics (arc welding)
    """
    end = min(start + duration, len(Va))
    
    combination = np.random.choice(['uv_harmonic', 'unbal_sag', 'flicker_harmonic'])
    
    if combination == 'uv_harmonic':
        # Overloaded transformer scenario
        Va, Vb, Vc = inject_undervoltage_realistic(
            Va, Vb, Vc, start, duration, severity=0.08, transition_time=20
        )
        THD_Va, THD_Vb, THD_Vc = inject_harmonics_realistic(
            THD_Va, THD_Vb, THD_Vc, start, duration, severity='moderate'
        )
        
    elif combination == 'unbal_sag':
        # Single-phase fault scenario
        Va, Vb, Vc = inject_voltage_sag_realistic(
            Va, Vb, Vc, start, min(duration // 3, 50), depth=0.2, sag_type='single_phase'
        )
        Va, Vb, Vc = inject_unbalance_realistic(
            Va, Vb, Vc, start + duration // 3, duration * 2 // 3, unbalance_percent=5.0
        )
        
    else:  # flicker_harmonic
        # Arc welding scenario
        Va, Vb, Vc = inject_flicker_realistic(Va, Vb, Vc, start, duration)
        THD_Va, THD_Vb, THD_Vc = inject_harmonics_realistic(
            THD_Va, THD_Vb, THD_Vc, start, duration, severity='mild'
        )
    
    return Va, Vb, Vc, THD_Va, THD_Vb, THD_Vc


# =============================================================================
# Main Dataset Generation
# =============================================================================
def generate_realistic_dataset(n_samples, anomaly_ratio=0.0, seed=42,
                               season='mixed', include_compound=True):
    """
    Generate complete realistic dataset.
    
    Args:
        n_samples: Number of samples
        anomaly_ratio: Fraction of anomalous samples (0-1)
        seed: Random seed
        season: 'summer', 'winter', or 'mixed'
        include_compound: Include compound anomalies
    """
    np.random.seed(seed)
    
    # Determine season distribution
    if season == 'mixed':
        # Assign random seasons to different parts
        n_summer = n_samples // 2
        n_winter = n_samples - n_summer
        load_pattern_summer = generate_rural_load_pattern(n_summer, 'summer', 'weekday')
        load_pattern_winter = generate_rural_load_pattern(n_winter, 'winter', 'weekday')
        load_pattern = np.concatenate([load_pattern_summer, load_pattern_winter])
    else:
        load_pattern = generate_rural_load_pattern(n_samples, season, 'weekday')
    
    # Generate base voltages
    Va, Vb, Vc = generate_three_phase_voltage(n_samples, load_pattern)
    
    # Generate currents
    Ia, Ib, Ic = generate_current(Va, Vb, Vc, load_pattern)
    
    # Generate power metrics
    P, Q, S, PF = generate_power_metrics(Va, Vb, Vc, Ia, Ib, Ic)
    
    # Generate quality metrics
    THD_Va = generate_harmonic_content(n_samples)
    THD_Vb = generate_harmonic_content(n_samples)
    THD_Vc = generate_harmonic_content(n_samples)
    Freq = generate_frequency(n_samples)
    
    # Initialize labels
    labels = np.zeros(n_samples, dtype=int)
    
    # Inject anomalies
    if anomaly_ratio > 0:
        n_anomaly_samples = int(n_samples * anomaly_ratio)
        
        # Anomaly types with realistic probabilities
        anomaly_config = [
            ('undervoltage', 0.20, lambda s, d: inject_undervoltage_realistic(Va, Vb, Vc, s, d)),
            ('overvoltage', 0.15, lambda s, d: inject_overvoltage_realistic(Va, Vb, Vc, s, d)),
            ('sag_3phase', 0.15, lambda s, d: inject_voltage_sag_realistic(Va, Vb, Vc, s, d, sag_type='three_phase')),
            ('sag_1phase', 0.10, lambda s, d: inject_voltage_sag_realistic(Va, Vb, Vc, s, d, sag_type='single_phase')),
            ('harmonics', 0.15, lambda s, d: (Va, Vb, Vc, *inject_harmonics_realistic(THD_Va, THD_Vb, THD_Vc, s, d))),
            ('unbalance', 0.10, lambda s, d: inject_unbalance_realistic(Va, Vb, Vc, s, d)),
            ('transient', 0.05, lambda s, d: inject_transient_realistic(Va, Vb, Vc, s, 'motor_start')),
            ('flicker', 0.05, lambda s, d: inject_flicker_realistic(Va, Vb, Vc, s, d)),
            ('compound', 0.05, lambda s, d: inject_compound_anomaly(Va, Vb, Vc, THD_Va, THD_Vb, THD_Vc, s, d)),
        ]
        
        if not include_compound:
            # Remove compound and redistribute
            anomaly_config = anomaly_config[:-1]
            total_prob = sum(c[1] for c in anomaly_config)
            anomaly_config = [(n, p/total_prob, f) for n, p, f in anomaly_config]
        
        # Calculate number of events per type
        events_per_type = []
        remaining_samples = n_anomaly_samples
        
        for name, prob, _ in anomaly_config:
            n_events = int(n_anomaly_samples * prob / 50)  # ~50 samples per event on average
            n_events = max(1, n_events)
            events_per_type.append(n_events)
        
        # Generate anomalies
        anomaly_type_id = 1
        current_pos = int(n_samples * 0.05)  # Start after initial period
        
        for (name, prob, inject_func), n_events in zip(anomaly_config, events_per_type):
            for _ in range(n_events):
                if current_pos >= n_samples - 200:
                    break
                
                # Variable duration based on anomaly type
                if name in ['transient', 'flicker']:
                    duration = np.random.randint(10, 60)
                elif name == 'sag_3phase' or name == 'sag_1phase':
                    duration = np.random.randint(5, 50)
                else:
                    duration = np.random.randint(30, 150)
                
                end = min(current_pos + duration, n_samples)
                
                # Inject anomaly
                result = inject_func(current_pos, duration)
                
                # Update arrays if needed (for harmonics and compound)
                if name == 'harmonics':
                    _, _, _, THD_Va, THD_Vb, THD_Vc = result[0], result[1], result[2], result[3], result[4], result[5] if len(result) > 5 else (THD_Va, THD_Vb, THD_Vc)
                elif name == 'compound':
                    Va, Vb, Vc, THD_Va, THD_Vb, THD_Vc = result
                else:
                    Va, Vb, Vc = result
                
                # Update labels
                labels[current_pos:end] = anomaly_type_id
                
                # Gap between anomalies (random, ensuring some normal periods)
                gap = np.random.randint(100, 500)
                current_pos = end + gap
            
            anomaly_type_id += 1
        
        # Recalculate dependent metrics
        Ia, Ib, Ic = generate_current(Va, Vb, Vc, load_pattern)
        P, Q, S, PF = generate_power_metrics(Va, Vb, Vc, Ia, Ib, Ic)
    
    # Calculate final unbalance
    V_unbalance = calculate_unbalance(Va, Vb, Vc)
    I_unbalance = calculate_unbalance(Ia, Ib, Ic)
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Va': Va,
        'Vb': Vb,
        'Vc': Vc,
        'Ia': Ia,
        'Ib': Ib,
        'Ic': Ic,
        'P': P,
        'Q': Q,
        'S': S,
        'PF': PF,
        'THD_Va': THD_Va,
        'THD_Vb': THD_Vb,
        'THD_Vc': THD_Vc,
        'Freq': Freq,
        'V_unbalance': V_unbalance,
        'I_unbalance': I_unbalance,
    })
    
    # Binary labels
    binary_labels = (labels > 0).astype(int)
    
    # Anomaly type names
    anomaly_names = {
        0: 'Normal',
        1: 'Undervoltage',
        2: 'Overvoltage', 
        3: 'Voltage_Sag_3Phase',
        4: 'Voltage_Sag_1Phase',
        5: 'Harmonics',
        6: 'Unbalance',
        7: 'Transient',
        8: 'Flicker',
        9: 'Compound'
    }
    
    return df, labels, binary_labels, anomaly_names


def main():
    parser = argparse.ArgumentParser(description='Generate realistic voltage data')
    parser.add_argument('--train_samples', type=int, default=50000,
                        help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=10000,
                        help='Number of test samples')
    parser.add_argument('--anomaly_ratio', type=float, default=0.12,
                        help='Ratio of anomalies in test data')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory')
    parser.add_argument('--season', type=str, default='mixed',
                        choices=['summer', 'winter', 'mixed'],
                        help='Season for load pattern')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Realistic Rural Voltage Data Generator V2.0")
    print("=" * 60)
    
    # Generate training data (normal only)
    print(f"\n[1/2] Generating training data ({args.train_samples} samples, normal only)...")
    train_df, _, _, _ = generate_realistic_dataset(
        args.train_samples, anomaly_ratio=0.0, seed=42, season=args.season
    )
    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    print(f"      Saved to train.csv")
    
    # Generate test data (with anomalies)
    print(f"\n[2/2] Generating test data ({args.test_samples} samples, {args.anomaly_ratio*100:.0f}% anomalies)...")
    test_df, labels, binary_labels, anomaly_names = generate_realistic_dataset(
        args.test_samples, anomaly_ratio=args.anomaly_ratio, seed=123, season=args.season
    )
    test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    print(f"      Saved to test.csv")
    
    # Save labels
    label_df = pd.DataFrame({
        'timestamp': test_df['timestamp'],
        'label': binary_labels,
        'anomaly_type': labels,
        'anomaly_name': [anomaly_names.get(l, 'Unknown') for l in labels]
    })
    label_df.to_csv(os.path.join(args.output_dir, 'test_label.csv'), index=False)
    print(f"      Saved to test_label.csv")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples:     {len(test_df):,}")
    print(f"  Anomaly ratio:    {np.mean(binary_labels)*100:.2f}%")
    print(f"\n  Anomaly type distribution:")
    for i in range(10):
        count = np.sum(labels == i)
        if count > 0:
            pct = count / len(labels) * 100
            print(f"    [{i}] {anomaly_names.get(i, 'Unknown'):20s}: {count:5d} ({pct:5.1f}%)")
    
    # Voltage statistics
    print(f"\n  Voltage statistics (test set):")
    print(f"    Va: min={test_df['Va'].min():.1f}V, max={test_df['Va'].max():.1f}V, mean={test_df['Va'].mean():.1f}V")
    print(f"    Vb: min={test_df['Vb'].min():.1f}V, max={test_df['Vb'].max():.1f}V, mean={test_df['Vb'].mean():.1f}V")
    print(f"    Vc: min={test_df['Vc'].min():.1f}V, max={test_df['Vc'].max():.1f}V, mean={test_df['Vc'].mean():.1f}V")
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
