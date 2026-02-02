"""
Isolation Forest Wrapper for Time Series Anomaly Detection

A PyTorch-compatible wrapper for sklearn's Isolation Forest algorithm.
This model is used for unsupervised anomaly detection without reconstruction.

Note: This model doesn't use reconstruction loss. Instead, it uses the Isolation Forest
anomaly scores directly. The framework's test phase will need special handling.

Reference:
    Liu et al. (2008) "Isolation Forest"
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest as SklearnIsolationForest


class Model(nn.Module):
    """
    Isolation Forest wrapper for anomaly detection.
    
    Since Isolation Forest is not a reconstruction-based method,
    we use a simple identity mapping during training and cache
    the sklearn model for scoring during evaluation.
    
    Note: For proper evaluation, use the dedicated IsolationForestExperiment class.
    This wrapper provides framework compatibility for the standard training loop.
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Isolation Forest parameters
        self.n_estimators = getattr(configs, 'n_estimators', 100)
        self.max_samples = getattr(configs, 'max_samples', 'auto')
        self.contamination = getattr(configs, 'anomaly_ratio', 1.0) / 100.0  # Convert percentage to fraction
        self.random_state = getattr(configs, 'seed', 42)
        
        # Simple identity layer for reconstruction (framework compatibility)
        # This allows the model to pass through the training loop
        self.identity = nn.Identity()
        
        # The actual sklearn model will be stored here
        self._sklearn_model = None
        self._is_fitted = False
        
    @property
    def sklearn_model(self):
        """Lazy initialization of sklearn Isolation Forest."""
        if self._sklearn_model is None:
            self._sklearn_model = SklearnIsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                contamination=min(self.contamination, 0.5),  # sklearn limit
                random_state=self.random_state,
                n_jobs=-1
            )
        return self._sklearn_model
    
    def fit_sklearn(self, data):
        """
        Fit the sklearn Isolation Forest model.
        
        Args:
            data: numpy array of shape (n_samples, n_features)
        """
        # Flatten if needed: (n_windows, seq_len, features) -> (n_windows, seq_len * features)
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], -1)
        
        self.sklearn_model.fit(data)
        self._is_fitted = True
    
    def predict_sklearn(self, data):
        """
        Predict anomaly scores using sklearn model.
        
        Args:
            data: numpy array of shape (n_samples, n_features) or (n_samples, seq_len, features)
            
        Returns:
            scores: Anomaly scores (higher = more anomalous)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit_sklearn first.")
        
        # Flatten if needed
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], -1)
        
        # Get anomaly scores (negative of decision function, so higher = more anomalous)
        scores = -self.sklearn_model.decision_function(data)
        
        return scores
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass - returns identity for framework compatibility.
        
        For actual anomaly detection, use fit_sklearn and predict_sklearn methods.
        """
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        else:
            return self.anomaly_detection(x_enc)
    
    def anomaly_detection(self, x_enc):
        """
        Returns input unchanged for reconstruction loss computation.
        
        Note: For proper Isolation Forest evaluation, use the dedicated experiment class.
        """
        # Return identity - this makes reconstruction error = 0
        # The actual anomaly detection happens in the sklearn model
        return self.identity(x_enc)
