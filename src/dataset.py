"""
dataset.py
----------
Loads and preprocesses BRFSS datasets for temporal incremental learning.

TEMPORAL APPROACH:
    Instead of splitting one dataset by BMI category, we now treat each
    survey year as a separate task. The model learns chronologically:

        Task 1 = BRFSS 2015  (oldest — trained first)
        Task 2 = BRFSS 2019  (middle)
        Task 3 = BRFSS 2023  (most recent — trained last)

    This simulates real-world deployment where a clinical model must adapt
    to new patient data year by year without forgetting what it learned before.

WHY TEMPORAL SHIFT CREATES GENUINE DOMAIN SHIFT:
    Over 8 years, population-level health patterns change meaningfully:
        - Obesity rates have increased
        - COVID-19 introduced new comorbidities (post-2019)
        - Physical activity trends shifted
        - Socioeconomic conditions changed
        - Diagnostic criteria and reporting practices evolved

    These genuine changes mean each year's data has a different statistical
    distribution — exactly the scenario EWC is designed to handle.

SHARED FEATURES:
    BRFSS surveys use a consistent core structure across years. We identify
    the common features shared across all three datasets and use only those,
    ensuring the model sees the same input space for every task.

DATA PLACEMENT:
    Place your files as:
        data/brfss_2015.csv
        data/brfss_2019.csv
        data/brfss_2023.csv
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: imbalanced-learn not installed. Run: pip install imbalanced-learn")

# Target column name — the diabetes diagnosis variable
TARGET_COL = 'Diabetes_binary'

# Candidate names for the target column across different BRFSS versions
TARGET_CANDIDATES = [
    'Diabetes_binary', 'Diabetes', 'diabetes', 'DIABETE3', 'diabete3'
]

# Dataset file paths in chronological order
DATASET_PATHS = {
    '2015': './data/modified_diabetes_indicator_dataset_2015.csv',
    '2019': './data/modified_diabetes_indicator_dataset_2019.csv',
    '2023': './data/modified_diabetes_indicator_dataset_2023.csv',
}


# ──────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────

def _find_target_column(df, year):
    """Find the diabetes target column in a dataframe."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find target column in BRFSS {year}.\n"
        f"Tried: {TARGET_CANDIDATES}\n"
        f"Available columns: {list(df.columns)}"
    )


def _load_single_year(path, year):
    """
    Load one BRFSS CSV file, find the target column, and keep only
    numeric feature columns. Returns a cleaned dataframe with a
    standardised target column name (Diabetes_binary).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nBRFSS {year} not found at: {path}\n"
            f"Please place the CSV file at that path and try again."
        )

    df     = pd.read_csv(path)
    target = _find_target_column(df, year)

    # Keep only numeric columns (features + target)
    numeric_cols  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df            = df[numeric_cols].dropna().drop_duplicates()

    # Standardise target column name
    if target != TARGET_COL:
        df = df.rename(columns={target: TARGET_COL})

    diabetic_pct = df[TARGET_COL].mean() * 100
    print(f"  BRFSS {year}: {len(df):,} rows | {diabetic_pct:.1f}% diabetic | {len(df.columns)-1} features")

    return df


def _get_shared_features(dataframes):
    """
    Find feature columns that exist in ALL datasets.
    This ensures the model sees the same input space across all years.
    Excludes the target column.
    """
    feature_sets = [
        set(df.columns) - {TARGET_COL}
        for df in dataframes
    ]
    shared = sorted(set.intersection(*feature_sets))

    print(f"\n  Shared features across all years: {len(shared)}")
    print(f"  {shared}")
    return shared


def _make_task(df, feature_cols, scaler, apply_smote, task_num):
    """
    Prepare one task: scale features, train/test split, apply SMOTE.
    Scaler must already be fitted before calling this function.
    """
    X        = df[feature_cols].values
    y        = df[TARGET_COL].values
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if apply_smote and SMOTE_AVAILABLE:
        smote            = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    diabetic_pct = y_test.mean() * 100
    print(f"    Task {task_num}: {len(X_train):,} train (after SMOTE) | "
          f"{len(X_test):,} test | {diabetic_pct:.1f}% diabetic in test")

    return {
        'X_train': X_train,
        'X_test':  X_test,
        'y_train': y_train,
        'y_test':  y_test,
    }


# ──────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────

def load_temporal_tasks(apply_smote=True):
    """
    Load BRFSS 2015, 2019, and 2023 as three sequential temporal tasks.

    Each year is one task. The model trains on them in chronological order,
    with EWC protecting knowledge from previous years.

    Returns
    -------
    tasks      : list of 3 dicts, each with X_train, X_test, y_train, y_test
    scaler     : fitted StandardScaler (same scale across all tasks)
    task_names : list of human-readable task labels
    feature_cols: list of shared feature column names
    """
    print("\n  Loading BRFSS datasets in chronological order...")

    # Load all three years
    dataframes = {}
    for year, path in DATASET_PATHS.items():
        dataframes[year] = _load_single_year(path, year)

    # Find features shared across ALL years
    feature_cols = _get_shared_features(list(dataframes.values()))

    if len(feature_cols) == 0:
        raise ValueError(
            "No shared feature columns found across the three datasets. "
            "Check that all three CSV files use compatible column names."
        )

    # Fit the scaler on ALL data combined so scaling is consistent
    # This is important — if each year had its own scaler, the same
    # BMI value would be scaled differently in 2015 vs 2023
    all_X = pd.concat([
        df[feature_cols] for df in dataframes.values()
    ])
    scaler = StandardScaler()
    scaler.fit(all_X)
    print(f"\n  Scaler fitted on combined data ({len(all_X):,} rows)")

    # Build tasks in chronological order
    task_names = []
    tasks      = []

    print("\n  Building tasks:")
    for i, (year, df) in enumerate(dataframes.items()):
        task_names.append(f"Task {i+1} — BRFSS {year}")
        task = _make_task(df, feature_cols, scaler, apply_smote, i + 1)
        tasks.append(task)

    return tasks, scaler, task_names, feature_cols


def to_dataloader(X, y, batch_size=64, shuffle=True):
    """Convert numpy arrays to a PyTorch DataLoader."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
