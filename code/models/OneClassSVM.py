"""
One-Class SVM Wrapper for Time Series Anomaly Detection

A PyTorch-compatible wrapper for sklearn's One-Class SVM algorithm.
This model learns a decision boundary around normal data.

Note: This model doesn't use reconstruction loss. Instead, it uses the SVM
decision function directly.

Reference:
    Scholkopf et al. (2001) "Estimating the Support of a High-Dimensional Distribution"
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import OneClassSVM as SklearnOneClassSVM
from sklearn.preprocessing import StandardScaler


class Model(nn.Module):
    """
    One-Class SVM wrapper for anomaly detection.
    
    Since One-Class SVM is not a reconstruction-based method,
    we use a simple identity mapping during training and cache
    the sklearn model for scoring during evaluation.
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # One-Class SVM parameters
        self.kernel = getattr(configs, 'svm_kernel', 'rbf')
        self.gamma = getattr(configs, 'svm_gamma', 'scale')
        self.nu = getattr(configs, 'anomaly_ratio', 1.0) / 100.0  # Anomaly fraction
        self.nu = max(0.001, min(self.nu, 0.5))  # sklearn constraint: 0 < nu <= 0.5
        
        # Simple identity layer for reconstruction (framework compatibility)
        self.identity = nn.Identity()
        
        # The actual sklearn model will be stored here
        self._sklearn_model = None
        self._scaler = None
        self._is_fitted = False
        
    @property
    def sklearn_model(self):
        """Lazy initialization of sklearn One-Class SVM."""
        if self._sklearn_model is None:
            self._sklearn_model = SklearnOneClassSVM(
                kernel=self.kernel,
                gamma=self.gamma,
                nu=self.nu,
                shrinking=True,
                cache_size=500
            )
        return self._sklearn_model
    
    def fit_sklearn(self, data, max_samples=10000):
        """
        Fit the sklearn One-Class SVM model.
        
        Args:
            data: numpy array of shape (n_samples, n_features) or (n_samples, seq_len, features)
            max_samples: Maximum samples for fitting (SVM is memory intensive)
        """
        # Flatten if needed: (n_windows, seq_len, features) -> (n_windows, seq_len * features)
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], -1)
        
        # Subsample if too large (SVM is memory intensive)
        if data.shape[0] > max_samples:
            indices = np.random.choice(data.shape[0], max_samples, replace=False)
            data = data[indices]
        
        # Scale data (important for SVM with RBF kernel)
        self._scaler = StandardScaler()
        data_scaled = self._scaler.fit_transform(data)
        
        # Fit model
        self.sklearn_model.fit(data_scaled)
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
        
        # Scale data
        data_scaled = self._scaler.transform(data)
        
        # Get anomaly scores (negative of decision function, so higher = more anomalous)
        scores = -self.sklearn_model.decision_function(data_scaled)
        
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
        
        Note: For proper One-Class SVM evaluation, use the dedicated experiment class.
        """
        return self.identity(x_enc)
