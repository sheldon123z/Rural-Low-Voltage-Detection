"""
Data Factory for Voltage Anomaly Detection
Standalone version - independent from main TSLib
"""

from data_provider.data_loader import (
    PSMSegLoader,
    MSLSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
    RuralVoltageSegLoader
)
from torch.utils.data import DataLoader


# Dataset registry
data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'RuralVoltage': RuralVoltageSegLoader,
}


def data_provider(args, flag):
    """
    Create data loader for anomaly detection task.
    
    Args:
        args: Arguments containing data configuration
        flag: 'train', 'val', or 'test'
        
    Returns:
        data_set: Dataset object
        data_loader: DataLoader object
    """
    Data = data_dict[args.data]
    
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    data_set = Data(
        args=args,
        root_path=args.root_path,
        win_size=args.seq_len,
        flag=flag,
    )
    
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader


def get_available_datasets():
    """Return list of available dataset names."""
    return list(data_dict.keys())
