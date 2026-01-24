"""
Data Loaders for Voltage Anomaly Detection
Standalone version - independent from main TSLib

Supports multiple anomaly detection datasets:
- PSM: Server Machine Dataset
- MSL: Mars Science Laboratory
- SMAP: Soil Moisture Active Passive
- SMD: Server Machine Dataset
- SWAT: Secure Water Treatment
- RuralVoltage: Rural Power Grid Voltage (custom)
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class PSMSegLoader(Dataset):
    """
    PSM (Pool Server Metrics) Dataset Loader.

    Data structure:
        dataset/PSM/
        ├── train.csv
        ├── test.csv
        └── test_label.csv
    """

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            raise FileNotFoundError(
                f"PSM dataset not found in {root_path}. "
                f"Please ensure train.csv, test.csv, and test_label.csv exist."
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_label_df = pd.read_csv(label_path)

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8) :]
        self.test_labels = test_label_df.values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class MSLSegLoader(Dataset):
    """
    MSL (Mars Science Laboratory) Dataset Loader.

    Data structure:
        dataset/MSL/
        ├── MSL_train.npy
        ├── MSL_test.npy
        └── MSL_test_label.npy
    """

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            raise FileNotFoundError(
                f"MSL dataset not found in {root_path}. "
                f"Please ensure MSL_train.npy, MSL_test.npy, and MSL_test_label.npy exist."
            )

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_label = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8) :]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMAPSegLoader(Dataset):
    """
    SMAP (Soil Moisture Active Passive) Dataset Loader.

    Data structure:
        dataset/SMAP/
        ├── SMAP_train.npy
        ├── SMAP_test.npy
        └── SMAP_test_label.npy
    """

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            raise FileNotFoundError(
                f"SMAP dataset not found in {root_path}. "
                f"Please ensure SMAP_train.npy, SMAP_test.npy, and SMAP_test_label.npy exist."
            )

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_label = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8) :]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMDSegLoader(Dataset):
    """
    SMD (Server Machine Dataset) Loader.

    Data structure:
        dataset/SMD/
        ├── SMD_train.npy
        ├── SMD_test.npy
        └── SMD_test_label.npy
    """

    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            raise FileNotFoundError(
                f"SMD dataset not found in {root_path}. "
                f"Please ensure SMD_train.npy, SMD_test.npy, and SMD_test_label.npy exist."
            )

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_label = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8) :]
        self.test_labels = test_label

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SWATSegLoader(Dataset):
    """
    SWAT (Secure Water Treatment) Dataset Loader.

    Data structure:
        dataset/SWAT/
        ├── swat_train2.csv
        └── swat2.csv (test with labels in last column)
    """

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path = os.path.join(root_path, "swat2.csv")

        if not all(os.path.exists(p) for p in [train2_path, test_path]):
            raise FileNotFoundError(
                f"SWAT dataset not found in {root_path}. "
                f"Please ensure swat_train2.csv and swat2.csv exist."
            )

        train_data = pd.read_csv(train2_path)
        test_data = pd.read_csv(test_path)

        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8) :]
        self.test_labels = labels

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class RuralVoltageSegLoader(Dataset):
    """
    Rural Power Grid Voltage Anomaly Detection Dataset Loader.

    Designed specifically for power grid voltage monitoring with features:
    - Three-phase voltage: Va, Vb, Vc (V)
    - Three-phase current: Ia, Ib, Ic (A)
    - Power metrics: P, Q, S, PF
    - Power quality: THD_Va, THD_Vb, THD_Vc
    - Frequency and unbalance: Freq, V_unbalance, I_unbalance

    Data structure:
        dataset/RuralVoltage/
        ├── train.csv        # Training data (mainly normal operation)
        ├── test.csv         # Test data (with anomalies)
        ├── test_label.csv   # Test labels
        └── metadata.json    # Meta information (optional)

    Label mapping for multi-class anomaly detection:
        0: Normal           - Normal operation
        1: Undervoltage     - Low voltage (V < 198V, -10%)
        2: Overvoltage      - Over voltage (V > 242V, +10%)
        3: Voltage_Sag      - Voltage sag (transient drop)
        4: Harmonic         - Harmonic distortion (THD > 5%)
        5: Unbalance        - Three-phase unbalance (> 4%)

    Voltage thresholds based on China GB/T 12325-2008:
        - Nominal voltage: 220V
        - Lower limit: 198V (-10%)
        - Upper limit: 242V (+10%)
        - THD limit: 5%
        - Unbalance limit: 4%
    """

    # Label mapping
    LABEL_MAPPING = {
        0: "Normal",
        1: "Undervoltage",
        2: "Overvoltage",
        3: "Voltage_Sag",
        4: "Harmonic",
        5: "Unbalance",
    }

    # Voltage thresholds (China GB/T 12325-2008)
    VOLTAGE_NOMINAL = 220.0
    VOLTAGE_LOWER_LIMIT = 198.0  # -10%
    VOLTAGE_UPPER_LIMIT = 242.0  # +10%
    THD_LIMIT = 5.0  # %
    UNBALANCE_LIMIT = 4.0  # %

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.root_path = root_path

        # File paths
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        # Load data
        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            raise FileNotFoundError(
                f"RuralVoltage dataset not found in {root_path}. "
                f"Please ensure train.csv, test.csv, and test_label.csv exist."
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_label_df = pd.read_csv(label_path)

        # Extract features (exclude timestamp column if present)
        feature_cols = [
            col
            for col in train_df.columns
            if col not in ["timestamp", "date", "time", "label"]
        ]

        # Process training data
        train_data = train_df[feature_cols].values
        train_data = np.nan_to_num(train_data, nan=0.0)

        # Fit scaler on training data
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        # Process test data
        test_data = test_df[feature_cols].values
        test_data = np.nan_to_num(test_data, nan=0.0)
        test_data = self.scaler.transform(test_data)

        # Process labels
        if "label" in test_label_df.columns:
            test_labels = test_label_df["label"].values.reshape(-1, 1)
        else:
            label_cols = [
                col
                for col in test_label_df.columns
                if col not in ["timestamp", "date", "time"]
            ]
            test_labels = test_label_df[label_cols[0]].values.reshape(-1, 1)

        self.train = train_data
        self.test = test_data
        self.test_labels = test_labels

        # Create validation set from training data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8) :]

        # Store feature names for reference
        self.feature_names = feature_cols
        self.num_features = len(feature_cols)

        print(f"RuralVoltage Dataset loaded:")
        print(f"  - Features: {self.num_features} ({', '.join(feature_cols[:5])}...)")
        print(f"  - Train: {self.train.shape}")
        print(f"  - Test: {self.test.shape}")
        print(f"  - Val: {self.val.shape}")

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )

    def get_voltage_statistics(self):
        """Get voltage-specific statistics for analysis."""
        voltage_indices = [
            i for i, name in enumerate(self.feature_names) if name in ["Va", "Vb", "Vc"]
        ]
        if voltage_indices:
            train_actual = self.scaler.inverse_transform(self.train)
            voltage_data = train_actual[:, voltage_indices]
            return {
                "mean": np.mean(voltage_data, axis=0),
                "std": np.std(voltage_data, axis=0),
                "min": np.min(voltage_data, axis=0),
                "max": np.max(voltage_data, axis=0),
            }
        return None

    def get_anomaly_distribution(self):
        """Get distribution of anomaly labels in test set."""
        unique, counts = np.unique(self.test_labels, return_counts=True)
        distribution = {}
        for u, c in zip(unique, counts):
            label_name = self.LABEL_MAPPING.get(int(u), f"Unknown_{u}")
            distribution[label_name] = int(c)
        return distribution
