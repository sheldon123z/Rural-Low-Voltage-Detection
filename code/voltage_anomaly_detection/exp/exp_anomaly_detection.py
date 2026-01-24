"""
Anomaly Detection Experiment
Standalone version for Voltage Anomaly Detection

This module handles:
- Model training with reconstruction loss
- Validation with early stopping
- Testing with anomaly score computation
- Point adjustment evaluation
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from models import get_model
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from utils.print_args import print_args

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection:
    """
    Experiment class for anomaly detection using reconstruction-based approach.
    
    The model learns to reconstruct normal patterns during training.
    During testing, high reconstruction error indicates anomaly.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _acquire_device(self):
        """Select computing device (GPU/CPU)."""
        if self.args.use_gpu:
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Use GPU: cuda:{self.args.gpu}')
            else:
                print('GPU not available, using CPU')
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        """Build the model from configuration."""
        model = get_model(self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        """Get data loader for specified flag."""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        """Select optimizer."""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        """Select loss criterion (MSE for reconstruction)."""
        criterion = nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        """
        Validation step.
        
        Returns:
            total_loss: Average validation loss
        """
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                # Reconstruct
                outputs = self.model(batch_x, None, None, None)
                
                # Compute loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        """
        Train the anomaly detection model.
        
        Args:
            setting: Experiment identifier string
            
        Returns:
            model: Trained model
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # Create checkpoint path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                # Forward: reconstruct input
                outputs = self.model(batch_x, None, None, None)
                
                # Loss computation
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())
                
                # Logging
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                # Backward
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Early stopping check
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # Adjust learning rate
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        # Load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self, setting, test=0):
        """
        Test the anomaly detection model.
        
        Args:
            setting: Experiment identifier string
            test: Whether to load pretrained model (1) or use current model (0)
            
        Returns:
            None (prints metrics and saves results)
        """
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            ))
        
        # Create results folder
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Anomaly criterion (for determining threshold)
        attens_energy = []
        
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        # Phase 1: Get anomaly scores from training set (to set threshold)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                
                # Reconstruct
                outputs = self.model(batch_x, None, None, None)
                
                # Calculate reconstruction error per sample
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
        # Phase 2: Get anomaly scores from test set
        test_labels = []
        attens_energy = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                
                # Reconstruct
                outputs = self.model(batch_x, None, None, None)
                
                # Calculate reconstruction error
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        
        # Combine train and test energy for threshold computation
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        
        # Compute threshold using anomaly_ratio
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold:", threshold)
        
        # Predict anomalies
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)
        
        # Apply point adjustment
        gt, pred = adjustment(gt, pred)
        
        pred = np.array(pred)
        gt = np.array(gt)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        
        # Compute metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary'
        )
        
        print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}".format(
            accuracy, precision, recall, f_score))
        
        # Save results
        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}\n".format(
            accuracy, precision, recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # Save predictions and scores
        np.save(folder_path + 'pred.npy', pred)
        np.save(folder_path + 'gt.npy', gt)
        np.save(folder_path + 'test_energy.npy', test_energy)
        np.save(folder_path + 'threshold.npy', np.array([threshold]))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f_score,
            'threshold': threshold
        }
