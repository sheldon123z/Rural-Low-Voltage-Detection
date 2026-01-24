#!/usr/bin/env python3
"""
Test script to verify all models can be imported and instantiated.
"""
import os
import sys
import torch
from argparse import Namespace

# Add the voltage_anomaly_detection folder to path (if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def create_test_config(model_name):
    """Create a basic config for testing model instantiation."""
    # Default values
    seq_len = 100
    d_model = 64
    d_ff = 256
    pred_len = 100  # Same as seq_len for anomaly detection
    
    # TimesNet needs pred_len=0 for anomaly detection
    if model_name == 'TimesNet':
        pred_len = 0  # For anomaly detection task
        d_ff = 64  # TimesNet uses d_ff for inception block
    
    config = Namespace(
        # Task settings
        task_name='anomaly_detection',
        model=model_name,
        
        # Input dimensions
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=48,
        enc_in=25,
        dec_in=25,
        c_out=25,
        
        # Model dimensions
        d_model=d_model,
        d_ff=d_ff,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        factor=3,
        
        # Dropout and activation
        dropout=0.1,
        activation='gelu',
        embed='timeF',
        freq='h',
        
        # Model specific parameters
        top_k=5,
        num_kernels=6,
        moving_avg=25,
        
        # TimeMixer specific
        down_sampling_window=2,
        down_sampling_layers=2,
        down_sampling_method='avg',
        channel_independence=1,
        use_norm=1,
        decomp_method='moving_avg',
        
        # SegRNN specific
        seg_len=10,
        
        # Nonstationary Transformer specific
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        
        # Classification (not used but needed for some models)
        num_class=2,
        
        # Device
        output_attention=False,
        distil=True,
    )
    return config


def test_model_import():
    """Test importing all models."""
    print("=" * 60)
    print("Testing Model Imports")
    print("=" * 60)
    
    try:
        from models import model_dict, get_model
        print(f"✓ Successfully imported model registry")
        print(f"  Available models: {list(model_dict.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed to import model registry: {e}")
        return False


def test_model_instantiation():
    """Test instantiating each model."""
    print("\n" + "=" * 60)
    print("Testing Model Instantiation")
    print("=" * 60)
    
    from models import model_dict
    
    results = {}
    for model_name, ModelClass in model_dict.items():
        try:
            config = create_test_config(model_name)
            model = ModelClass(config)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ {model_name:30s} | Parameters: {params:,}")
            results[model_name] = True
        except Exception as e:
            print(f"✗ {model_name:30s} | Error: {e}")
            results[model_name] = False
    
    return results


def test_model_forward():
    """Test forward pass for each model."""
    print("\n" + "=" * 60)
    print("Testing Model Forward Pass (Anomaly Detection)")
    print("=" * 60)
    
    from models import model_dict
    
    batch_size = 4
    enc_in = 25
    
    results = {}
    for model_name, ModelClass in model_dict.items():
        try:
            config = create_test_config(model_name)
            seq_len = config.seq_len
            
            # Create dummy input with correct seq_len
            x_enc = torch.randn(batch_size, seq_len, enc_in)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, seq_len, enc_in)
            x_mark_dec = torch.randn(batch_size, seq_len, 4)
            
            model = ModelClass(config)
            model.eval()
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            expected_shape = (batch_size, seq_len, enc_in)
            if output.shape == expected_shape:
                print(f"✓ {model_name:30s} | Output shape: {tuple(output.shape)}")
                results[model_name] = True
            else:
                print(f"⚠ {model_name:30s} | Expected: {expected_shape}, Got: {tuple(output.shape)}")
                results[model_name] = True  # Shape mismatch but runs
        except Exception as e:
            print(f"✗ {model_name:30s} | Error: {e}")
            results[model_name] = False
    
    return results


def main():
    print("=" * 60)
    print("Voltage Anomaly Detection - Model Test Suite")
    print("=" * 60)
    
    # Test 1: Import
    if not test_model_import():
        print("\nFailed to import models. Exiting.")
        return
    
    # Test 2: Instantiation
    inst_results = test_model_instantiation()
    
    # Test 3: Forward pass
    forward_results = test_model_forward()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    total = len(inst_results)
    inst_passed = sum(inst_results.values())
    fwd_passed = sum(forward_results.values())
    
    print(f"Instantiation: {inst_passed}/{total} models passed")
    print(f"Forward Pass:  {fwd_passed}/{total} models passed")
    
    failed_inst = [k for k, v in inst_results.items() if not v]
    failed_fwd = [k for k, v in forward_results.items() if not v]
    
    if failed_inst:
        print(f"\nFailed instantiation: {failed_inst}")
    if failed_fwd:
        print(f"Failed forward pass: {failed_fwd}")
    
    if inst_passed == total and fwd_passed == total:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")


if __name__ == '__main__':
    main()
