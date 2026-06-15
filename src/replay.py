"""
replay.py
---------
Experience Replay Buffer for continual learning.

WHY REPLAY ALONGSIDE EWC:
    EWC protects weights by penalising changes. Replay protects knowledge
    by re-exposing the model to real examples from previous tasks during
    future training. The two methods are complementary:

        EWC    → protects the weight space (parameter-level protection)
        Replay → protects the data space   (sample-level protection)

    Using both together consistently outperforms either alone, especially
    when the domain shift between tasks is subtle (as with BRFSS years).

WHY 2000 SAMPLES (NOT 500):
    The previous experiment used 500 samples per past task. With training
    sets of 300,000 samples, 500 replay samples is only 0.17% of the data
    — completely drowned out by the new task. 2000 samples (0.67%) gives
    the replay signal enough weight to make a meaningful difference.

    The replay samples are mixed into each training batch at a 3:1 ratio
    (new data : replay data). This is enough to remind the model of past
    patterns without slowing down learning on the new task.

HOW IT WORKS:
    After training on Task N, a random subset of Task N training data
    is stored in the buffer. When training on Task N+1, each batch
    is augmented with a proportion of replay samples drawn randomly
    from all stored past tasks.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Stores samples from past tasks and provides them for replay
    during future task training.

    Args:
        samples_per_task : how many samples to store per past task
        replay_ratio     : fraction of each batch to fill with replay samples
                           e.g. 0.25 means 25% of each batch is replay data
    """

    def __init__(self, samples_per_task=2000, replay_ratio=0.25):
        self.samples_per_task = samples_per_task
        self.replay_ratio     = replay_ratio
        self.buffer_X         = []   # list of numpy arrays, one per past task
        self.buffer_y         = []   # list of numpy arrays, one per past task

    def add_task(self, X_train, y_train):
        """
        Store a random subset of samples from a completed task.

        Called after training on each task — before moving to the next.
        Stores at most samples_per_task samples per task.
        """
        n = min(self.samples_per_task, len(X_train))
        idx = np.random.choice(len(X_train), size=n, replace=False)
        self.buffer_X.append(X_train[idx])
        self.buffer_y.append(y_train[idx])
        print(f"  Replay buffer: stored {n} samples | "
              f"total stored tasks: {len(self.buffer_X)}")

    def get_replay_batch(self, batch_size):
        """
        Draw a random batch of replay samples from all stored past tasks.
        Returns None if the buffer is empty.
        """
        if not self.buffer_X:
            return None

        # Combine all stored past tasks
        all_X = np.concatenate(self.buffer_X, axis=0)
        all_y = np.concatenate(self.buffer_y, axis=0)

        n   = min(batch_size, len(all_X))
        idx = np.random.choice(len(all_X), size=n, replace=False)

        X = torch.tensor(all_X[idx], dtype=torch.float32)
        y = torch.tensor(all_y[idx], dtype=torch.float32)
        return X, y

    def replay_batch_size(self, current_batch_size):
        """
        How many replay samples to add to a batch of current_batch_size.
        Based on replay_ratio — default 25% of batch is replay data.
        """
        return max(1, int(current_batch_size * self.replay_ratio))

    @property
    def is_empty(self):
        return len(self.buffer_X) == 0

    @property
    def total_stored(self):
        return sum(len(x) for x in self.buffer_X)
