# src/data.py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

def load_qm7_dataset(test_size=0.2, val_size=0.1, random_state=42):
    """
    Loads the QM7 dataset, scales features, splits into train/val/test, 
    and converts to torch tensors.
    """
    # Load dataset
    qm7 = fetch_openml(name="qm7", version=1, as_frame=True)
    X = qm7.data
    y = qm7.target

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Optional: reduce features if too many (PCA) or choose top columns
    # For now, just scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test first
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Split train/val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative, random_state=random_state
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_qm7_dataset()
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
