"""
Kaggle Power Quality Dataset Preparation Script

Converts PowerQualityDistributionDataset1.csv to anomaly detection format.

Dataset info:
- 11,998 records with 128 waveform samples each
- 5 classes: 1, 2, 3, 4, 5 (balanced)
- Class 3 is selected as "normal" (largest group)
- Other classes (1, 2, 4, 5) are "anomaly"

Output files:
- train.csv: Training data (normal samples only, 80% of Class 3)
- test.csv: Test data (20% Class 3 + all anomaly samples)
- test_label.csv: Binary labels (0=normal, 1=anomaly)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
NORMAL_CLASS = 3  # Class to use as "normal"
TRAIN_RATIO = 0.8  # 80% normal for training
RANDOM_SEED = 42

def prepare_dataset():
    """Convert Kaggle dataset to anomaly detection format."""

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load original dataset
    input_file = os.path.join(script_dir, "PowerQualityDistributionDataset1.csv")
    print(f"Loading: {input_file}")

    df = pd.read_csv(input_file, index_col=0)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]} ... {df.columns.tolist()[-5:]}")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col != 'output']
    X = df[feature_cols].values  # (11998, 128)
    y = df['output'].values  # (11998,)

    print(f"\nClass distribution:")
    for cls in sorted(np.unique(y)):
        count = np.sum(y == cls)
        print(f"  Class {cls}: {count} ({100*count/len(y):.1f}%)")

    # Split by class
    normal_mask = (y == NORMAL_CLASS)
    anomaly_mask = ~normal_mask

    X_normal = X[normal_mask]
    X_anomaly = X[anomaly_mask]

    print(f"\nNormal (Class {NORMAL_CLASS}): {len(X_normal)}")
    print(f"Anomaly (Other classes): {len(X_anomaly)}")

    # Split normal data into train/test
    X_normal_train, X_normal_test = train_test_split(
        X_normal,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED
    )

    print(f"\nTrain (normal only): {len(X_normal_train)}")
    print(f"Test normal: {len(X_normal_test)}")
    print(f"Test anomaly: {len(X_anomaly)}")

    # Create test set: normal_test + all anomaly
    X_test = np.vstack([X_normal_test, X_anomaly])
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)),  # 0 = normal
        np.ones(len(X_anomaly))         # 1 = anomaly
    ])

    # Shuffle test set
    np.random.seed(RANDOM_SEED)
    shuffle_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx]
    y_test = y_test[shuffle_idx]

    print(f"\nFinal test set: {len(X_test)}")
    print(f"  Normal: {np.sum(y_test == 0)} ({100*np.sum(y_test == 0)/len(y_test):.1f}%)")
    print(f"  Anomaly: {np.sum(y_test == 1)} ({100*np.sum(y_test == 1)/len(y_test):.1f}%)")

    # Convert to DataFrame format for saving
    # Feature column names: Col1, Col2, ..., Col128
    col_names = [f'Col{i+1}' for i in range(X_normal_train.shape[1])]

    # Save train.csv (features only, no labels)
    train_df = pd.DataFrame(X_normal_train, columns=col_names)
    train_file = os.path.join(script_dir, "train.csv")
    train_df.to_csv(train_file, index=False)
    print(f"\nSaved: {train_file} ({train_df.shape})")

    # Save test.csv (features only)
    test_df = pd.DataFrame(X_test, columns=col_names)
    test_file = os.path.join(script_dir, "test.csv")
    test_df.to_csv(test_file, index=False)
    print(f"Saved: {test_file} ({test_df.shape})")

    # Save test_label.csv (binary labels)
    label_df = pd.DataFrame(y_test.astype(int), columns=['label'])
    label_file = os.path.join(script_dir, "test_label.csv")
    label_df.to_csv(label_file, index=False)
    print(f"Saved: {label_file} ({label_df.shape})")

    # Summary
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print("="*50)
    print(f"Training samples: {len(train_df)} (100% normal)")
    print(f"Test samples: {len(test_df)}")
    print(f"  - Normal: {np.sum(y_test == 0)}")
    print(f"  - Anomaly: {np.sum(y_test == 1)}")
    print(f"Anomaly ratio in test: {100*np.sum(y_test == 1)/len(y_test):.2f}%")
    print(f"Feature dimensions: {X_normal_train.shape[1]}")

    return train_df, test_df, label_df


if __name__ == "__main__":
    prepare_dataset()
