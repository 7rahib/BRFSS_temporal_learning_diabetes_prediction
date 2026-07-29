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
    These CSVs are NOT pre-balanced — diabetes prevalence is only
    ~15-17% in every year. This file does not do any resampling
    itself (no SMOTE, no sampler) — imbalance is instead handled at
    training time via a weighted BCELoss in train.py, using the
    pos_weight computed by compute_pos_weight() below.

DATA PLACEMENT:
    data/modified_diabetes_indicator_dataset_2015.csv
    data/modified_diabetes_indicator_dataset_2019.csv
    data/modified_diabetes_indicator_dataset_2023.csv
"""

import os
import numpy as np
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


def _make_task(df, feature_cols, scaler, task_num, val_size=0.15):
    X        = df[feature_cols].values
    y        = df[TARGET_COL].values
    X_scaled = scaler.transform(X)

    # Held-out test set — untouched until final reporting
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Carve a validation split OUT OF THE TRAINING DATA (not the test set).
    # This is used in Phase D to pick each task's classification threshold.
    # Calibrating on the test set and then reporting metrics on that same
    # test set is a mild form of leakage — picking the threshold on val
    # and only ever touching test for the final numbers avoids that.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size,
        random_state=42, stratify=y_train_full
    )

    pct = y_test.mean() * 100
    print(f"    Task {task_num}: {len(X_train):,} train | {len(X_val):,} val | "
          f"{len(X_test):,} test | {pct:.1f}% diabetic")
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
    }


def load_temporal_tasks(val_size=0.15):
    """
    Load BRFSS 2015, 2019, 2023 as three sequential tasks.

    Each task gets a train / val / test split. `val_size` is the fraction
    of the *training* data (not the total) carved out for validation —
    used for threshold calibration in Phase D, kept separate from test.

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
        tasks.append(_make_task(df, feature_cols, scaler, i + 1, val_size))

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


def compute_pos_weight(y_train, mode='full'):
    """
    Compute the pos_weight used by the weighted BCELoss (train.py) from
    one task's training labels.

    mode='full' (default): (# negative) / (# positive) — the standard,
        most aggressive choice. This is what pushed recall from ~10% to
        ~82% at the default threshold in earlier runs.
    mode='sqrt': sqrt of the above — a softer push. Trades some of that
        recall gain back for more precision, if 'full' feels too
        aggressive for your use case. Try both and compare calibrated
        F1/precision/recall — neither is objectively "correct".

    WHY THIS IS NEEDED AT ALL:
        These BRFSS CSVs are NOT pre-balanced — diabetes prevalence is
        only ~15-17%. Without correcting for this, the model learns
        that predicting "no diabetes" every time already gets ~83-84%
        accuracy, so it has little incentive to learn the minority
        (diabetic) class — this shows up as very low recall at the
        default 0.5 threshold.

    Returns: a scalar torch.Tensor, ready to pass as the `pos_weight`
             argument of train.py's weighted_bce_loss().
    """
    y_train = np.asarray(y_train)
    n_pos   = float((y_train == 1).sum())
    n_neg   = float((y_train == 0).sum())
    ratio   = n_neg / max(n_pos, 1.0)

    if mode == 'sqrt':
        ratio = ratio ** 0.5
    elif mode != 'full':
        raise ValueError(f"Unknown pos_weight mode: {mode!r}. Use 'full' or 'sqrt'.")

    return torch.tensor(ratio, dtype=torch.float32)
