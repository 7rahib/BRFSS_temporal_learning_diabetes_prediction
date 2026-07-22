"""
dataset.py
----------
Loads BRFSS 2015, 2019, and 2023 as three sequential temporal tasks.

Each year is one task. The model trains chronologically:
    Task 1 = BRFSS 2015
    Task 2 = BRFSS 2019
    Task 3 = BRFSS 2023

SHARED FEATURES:
    Columns present in all three datasets are identified automatically.
    Only shared columns are used so input size is consistent across tasks.

CLASS IMBALANCE:
    The datasets provided in data/ already come pre-balanced with SMOTE,
    so no oversampling is done here — this file only loads, cleans,
    scales, and splits the data.

DATA PLACEMENT:
    data/modified_diabetes_indicator_dataset_2015.csv
    data/modified_diabetes_indicator_dataset_2019.csv
    data/modified_diabetes_indicator_dataset_2023.csv
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TARGET_COL         = 'Diabetes_binary'
TARGET_CANDIDATES  = ['Diabetes_binary', 'Diabetes', 'diabetes', 'DIABETE3', 'diabete3']
DATASET_PATHS      = {
    '2015': 'data/modified_diabetes_indicator_dataset_2015.csv',
    '2019': 'data/modified_diabetes_indicator_dataset_2019.csv',
    '2023': 'data/modified_diabetes_indicator_dataset_2023.csv',
}


def _find_target(df, year):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No target column found in BRFSS {year}. Columns: {list(df.columns)}")


def _load_year(path, year):
    if not os.path.exists(path):
        raise FileNotFoundError(f"BRFSS {year} not found at: {path}")
    df     = pd.read_csv(path)
    target = _find_target(df, year)
    nums   = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df     = df[nums].dropna().drop_duplicates()
    if target != TARGET_COL:
        df = df.rename(columns={target: TARGET_COL})
    pct = df[TARGET_COL].mean() * 100
    print(f"  BRFSS {year}: {len(df):,} rows | {pct:.1f}% diabetic | {len(df.columns)-1} features")
    return df


def _shared_features(dataframes):
    sets   = [set(df.columns) - {TARGET_COL} for df in dataframes]
    shared = sorted(set.intersection(*sets))
    print(f"\n  Shared features across all years: {len(shared)}")
    return shared


def _make_task(df, feature_cols, scaler, task_num):
    X        = df[feature_cols].values
    y        = df[TARGET_COL].values
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    pct = y_test.mean() * 100
    print(f"    Task {task_num}: {len(X_train):,} train | {len(X_test):,} test | {pct:.1f}% diabetic")
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def load_temporal_tasks():
    """
    Load BRFSS 2015, 2019, 2023 as three sequential tasks.

    Returns: tasks, scaler, task_names, feature_cols
    """
    print("\n  Loading BRFSS datasets...")
    dataframes = {year: _load_year(path, year) for year, path in DATASET_PATHS.items()}

    feature_cols = _shared_features(list(dataframes.values()))
    if not feature_cols:
        raise ValueError("No shared feature columns found across datasets.")

    # Fit scaler on all years combined for consistent scaling
    all_X  = pd.concat([df[feature_cols] for df in dataframes.values()])
    scaler = StandardScaler()
    scaler.fit(all_X)
    print(f"\n  Scaler fitted on {len(all_X):,} combined rows")

    tasks      = []
    task_names = []
    print("\n  Building tasks:")
    for i, (year, df) in enumerate(dataframes.items()):
        task_names.append(f"Task {i+1} — BRFSS {year}")
        tasks.append(_make_task(df, feature_cols, scaler, i + 1))

    return tasks, scaler, task_names, feature_cols


def to_dataloader(X, y, batch_size=64, shuffle=True):
    """Convert numpy arrays to a PyTorch DataLoader."""
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
