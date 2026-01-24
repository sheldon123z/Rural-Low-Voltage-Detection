#!/usr/bin/env python3
"""
Voltage Anomaly Detection - Main Entry Point

Standalone anomaly detection module for rural power grid voltage monitoring.
Based on TimesNet (ICLR 2023) with reconstruction-based approach.

Usage:
    python run.py --model TimesNet --data PSM --is_training 1
    python run.py --model TimesNet --data RuralVoltage --root_path ./dataset/RuralVoltage/
    
Author: Time-Series-Library based module
"""

import argparse
import os
import random
import numpy as np
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.print_args import print_args


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Voltage Anomaly Detection')
    
    # Basic Config
    parser.add_argument('--is_training', type=int, required=True, default=1, 
                        help='status: 1=train, 0=test only')
    parser.add_argument('--model_id', type=str, default='test', 
                        help='model identifier')
    parser.add_argument('--model', type=str, default='TimesNet',
                        help='model name: TimesNet (more models coming soon)')
    
    # Data Config
    parser.add_argument('--data', type=str, default='PSM', 
                        help='dataset: PSM, MSL, SMAP, SMD, SWAT, RuralVoltage')
    parser.add_argument('--root_path', type=str, default='./dataset/PSM/', 
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.csv',
                        help='data file name (for custom datasets)')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate, S: univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature (S or MS tasks)')
    parser.add_argument('--freq', type=str, default='h',
                        help='time features encoding: s/t/h/d/b/w/m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    
    # Anomaly Detection Config
    parser.add_argument('--anomaly_ratio', type=float, default=1.0,
                        help='prior anomaly ratio in percentage (default: 1%)')
    
    # Model Architecture Config
    parser.add_argument('--seq_len', type=int, default=100,
                        help='input sequence length (window size)')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length (not used in anomaly detection)')
    parser.add_argument('--pred_len', type=int, default=0,
                        help='prediction length (0 for anomaly detection)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='TimesNet: number of top-k frequencies')
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='TimesNet: number of kernels in Inception block')
    parser.add_argument('--enc_in', type=int, default=25,
                        help='encoder input dimension (number of features)')
    parser.add_argument('--dec_in', type=int, default=25,
                        help='decoder input dimension')
    parser.add_argument('--c_out', type=int, default=25,
                        help='output dimension')
    parser.add_argument('--d_model', type=int, default=64,
                        help='model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64,
                        help='dimension of feed-forward network')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1,
                        help='attention factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding: timeF, fixed, learned')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation function')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention weights')
    
    # Training Config
    parser.add_argument('--num_workers', type=int, default=10,
                        help='data loader workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='number of experiments')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--des', type=str, default='test',
                        help='experiment description')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='learning rate adjustment type')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use automatic mixed precision training')
    
    # GPU Config
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='use GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='GPU device ids for multi-GPU')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021,
                        help='random seed')
    
    args = parser.parse_args()
    
    # Set task_name for model (must be set before print_args)
    args.task_name = 'anomaly_detection'
    
    # Set seed
    set_seed(args.seed)
    
    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.device_ids = [int(id_) for id_ in args.devices.replace(' ', '').split(',')]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print_args(args)
    
    # Run experiments
    for ii in range(args.itr):
        # Setting string for experiment identification
        setting = '{}_{}_{}_{}_ft{}_sl{}_dm{}_nh{}_el{}_df{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.task_name,
            args.features,
            args.seq_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.des,
            ii
        )
        
        exp = Exp_Anomaly_Detection(args)
        
        if args.is_training:
            print(f'>>>>>>>>>> Training: {setting} <<<<<<<<<<<<')
            exp.train(setting)
            
            print(f'>>>>>>>>>> Testing: {setting} <<<<<<<<<<<<')
            exp.test(setting)
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
        else:
            print(f'>>>>>>>>>> Testing: {setting} <<<<<<<<<<<<')
            exp.test(setting, test=1)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
