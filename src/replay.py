"""
replay.py
---------
Experience Replay Buffer for Continual Learning.

WHAT REPLAY DOES IN ONE SENTENCE:
    After training on each year's data, store a small random sample of that
    year's examples. When training on the next year, mix in those stored
    samples so the model sees old data again — preventing forgetting directly.

HOW IT COMBINES WITH EWC:
    EWC protects weights mathematically (via Fisher penalty).
    Replay protects knowledge empirically (by re-exposing the model to old data).
    Together they are stronger than either alone:
        - Replay prevents forgetting on the stored samples directly.
        - EWC generalises protection to the full distribution via Fisher scores.

THE BUFFER:
    Each task contributes up to `samples_per_task` examples (randomly chosen).
    Total buffer size = samples_per_task × number_of_past_tasks.
    Default 500 per task → 1000 samples when training on Task 3.

USAGE PATTERN:
    buf = ReplayBuffer(samples_per_task=500)

    # After Task 1 training:
    buf.add_task("BRFSS 2015", X_train_2015, y_train_2015)

    # When training on Task 2:
    combined_loader = buf.get_combined_loader(X_train_2019, y_train_2019, batch_size=64)
    train_ewc_replay(model, combined_loader, ewc_objects, lambda_, ...)

    # After Task 2 training:
    buf.add_task("BRFSS 2019", X_train_2019, y_train_2019)
    # ... and so on
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:
    """
    Stores a random subset of each past task's training data.

    Args:
        samples_per_task : maximum samples to store per task (default 500)
        random_seed      : for reproducible sampling (default 42)
    """

    def __init__(self, samples_per_task=500, random_seed=42):
        self.samples_per_task = samples_per_task
        self.random_seed      = random_seed

        # Storage — each entry is a dict with 'name', 'X', 'y'
        self._tasks = []

        # Running RNG so each call to add_task gets independent randomness
        self._rng = np.random.default_rng(random_seed)

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def add_task(self, task_name, X_train, y_train):
        """
        Sample up to `samples_per_task` examples from a completed task
        and store them in the buffer.

        Args:
            task_name : human-readable label (used for print messages)
            X_train   : numpy array, shape (N, features)
            y_train   : numpy array, shape (N,)
        """
        n       = len(X_train)
        n_store = min(self.samples_per_task, n)

        # Random subset without replacement
        indices = self._rng.choice(n, size=n_store, replace=False)
        X_buf   = X_train[indices].copy()
        y_buf   = y_train[indices].copy()

        self._tasks.append({'name': task_name, 'X': X_buf, 'y': y_buf})

        diabetic_pct = y_buf.mean() * 100
        print(f"  [Replay] Stored {n_store} samples from '{task_name}' "
              f"({diabetic_pct:.1f}% diabetic)")

    def size(self):
        """Total number of samples currently in the buffer."""
        return sum(len(t['X']) for t in self._tasks)

    def num_tasks(self):
        """Number of past tasks stored."""
        return len(self._tasks)

    def is_empty(self):
        """True if no past tasks have been added yet."""
        return len(self._tasks) == 0

    def get_combined_loader(self, X_current, y_current, batch_size=64, shuffle=True):
        """
        Combine current task data with all replayed past data into one DataLoader.

        The combined dataset contains:
            - All samples from the current task (X_current, y_current)
            - All buffered samples from every past task

        The model trains on this combined dataset, so it sees old examples
        again alongside new ones — directly fighting catastrophic forgetting.

        Args:
            X_current : numpy array for the current task
            y_current : numpy array for the current task
            batch_size: DataLoader batch size
            shuffle   : shuffle the combined dataset (keep True for training)

        Returns:
            DataLoader over the combined dataset
        """
        if self.is_empty():
            # No past tasks yet — just wrap the current data
            return self._make_loader(X_current, y_current, batch_size, shuffle)

        # Stack current + all buffered tasks
        X_parts = [X_current] + [t['X'] for t in self._tasks]
        y_parts = [y_current] + [t['y'] for t in self._tasks]

        X_combined = np.concatenate(X_parts, axis=0)
        y_combined = np.concatenate(y_parts, axis=0)

        current_n = len(X_current)
        replay_n  = self.size()
        print(f"  [Replay] Combined loader: {current_n} current + {replay_n} replay "
              f"= {len(X_combined)} total samples")

        return self._make_loader(X_combined, y_combined, batch_size, shuffle)

    def summary(self):
        """Print a summary of what is stored in the buffer."""
        if self.is_empty():
            print("  [Replay] Buffer is empty.")
            return
        print(f"  [Replay] Buffer contents ({self.size()} total samples):")
        for t in self._tasks:
            diabetic_pct = t['y'].mean() * 100
            print(f"    '{t['name']}': {len(t['X'])} samples | {diabetic_pct:.1f}% diabetic")

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def _make_loader(X, y, batch_size, shuffle):
        """Convert numpy arrays to a PyTorch DataLoader."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
