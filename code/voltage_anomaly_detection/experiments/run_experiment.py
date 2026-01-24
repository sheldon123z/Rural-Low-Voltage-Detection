#!/usr/bin/env python3
"""
实验运行器 - 已弃用

⚠️ 此文件已弃用。创新模型已集成到主模型目录。

请使用以下方式运行实验：

1. 使用主运行器:
   cd ..
   python run.py --model TPATimesNet --data RuralVoltage --root_path ./dataset/RuralVoltage/ ...

2. 使用Shell脚本:
   bash scripts/RuralVoltage/TPATimesNet.sh
   bash scripts/RuralVoltage/MTSTimesNet.sh  
   bash scripts/RuralVoltage/HybridTimesNet.sh

3. 运行所有实验:
   bash scripts/run_all_experiments.sh

创新模型位置: ../models/ (TPATimesNet.py, MTSTimesNet.py, HybridTimesNet.py)
实验脚本位置: ../scripts/RuralVoltage/
实验计划文档: ../EXPERIMENT_PLAN.md
"""

import sys
import os

def main():
    print(__doc__)
    print("\n" + "="*70)
    print("此脚本已弃用。请使用上述方法运行实验。")
    print("="*70)
    sys.exit(1)

if __name__ == '__main__':
    main()



def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Exp_Experiment(Exp_Anomaly_Detection):
    """
    Extended experiment class that supports our innovative models.
    """
    
    def _build_model(self):
        """Build model - uses experimental models if specified."""
        model_name = self.args.model
        
        # Check if it's an experimental model
        if model_name in EXPERIMENT_MODELS:
            model = get_experiment_model(model_name, self.args)
        else:
            # Fall back to standard model loading
            from models import get_model
            model = get_model(self.args)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            import torch.nn as nn
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model


def main():
    parser = argparse.ArgumentParser(description='Rural Voltage Anomaly Detection Experiments')
    
    # Basic Config
    parser.add_argument('--is_training', type=int, default=1, 
                        help='status: 1=train, 0=test only')
    parser.add_argument('--model_id', type=str, default='experiment', 
                        help='model identifier')
    parser.add_argument('--model', type=str, default='TPATimesNet',
                        help='model name: TPATimesNet, MTSTimesNet, HybridTimesNet, or standard models')
    
    # Data Config
    parser.add_argument('--data', type=str, default='RuralVoltage', 
                        help='dataset name')
    parser.add_argument('--root_path', type=str, default='./dataset/RuralVoltageV2/', 
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.csv',
                        help='data file name')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate, S: univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature')
    parser.add_argument('--freq', type=str, default='h',
                        help='time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    
    # Anomaly Detection Config
    parser.add_argument('--anomaly_ratio', type=float, default=1.0,
                        help='prior anomaly ratio in percentage')
    
    # Model Architecture Config
    parser.add_argument('--seq_len', type=int, default=100,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=0,
                        help='prediction length')
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of top-k frequencies')
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='number of kernels in Inception block')
    parser.add_argument('--enc_in', type=int, default=17,
                        help='encoder input dimension')
    parser.add_argument('--dec_in', type=int, default=17,
                        help='decoder input dimension')
    parser.add_argument('--c_out', type=int, default=17,
                        help='output dimension')
    parser.add_argument('--d_model', type=int, default=64,
                        help='model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128,
                        help='dimension of feed-forward network')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1,
                        help='attention factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation function')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention')
    
    # Experimental Model Specific Config
    parser.add_argument('--voltage_indices', type=int, nargs='+', default=[0, 1, 2],
                        help='indices of voltage features (Va, Vb, Vc)')
    parser.add_argument('--preset_periods', type=int, nargs='+', default=None,
                        help='preset periods for HybridTimesNet')
    parser.add_argument('--use_domain_norm', action='store_true', default=False,
                        help='use domain-adaptive normalization')
    
    # Training Config
    parser.add_argument('--num_workers', type=int, default=10,
                        help='data loader workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='number of experiments')
    parser.add_argument('--train_epochs', type=int, default=20,
                        help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--des', type=str, default='experiment',
                        help='experiment description')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='learning rate adjustment')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use automatic mixed precision')
    
    # GPU Config
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='use GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='GPU device ids')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021,
                        help='random seed')
    
    args = parser.parse_args()
    
    # Set task_name
    args.task_name = 'anomaly_detection'
    
    # Set seed
    set_seed(args.seed)
    
    # Check GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.device_ids = [int(id_) for id_ in args.devices.replace(' ', '').split(',')]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print_args(args)
    
    # Run experiments
    for ii in range(args.itr):
        # Setting string for this experiment
        setting = '{}_{}_{}_{}_ft{}_sl{}_dm{}_nh{}_el{}_df{}_{}'.format(
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
            args.des
        )
        
        # Create experiment
        exp = Exp_Experiment(args)
        
        if args.is_training:
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
        else:
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
        
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
