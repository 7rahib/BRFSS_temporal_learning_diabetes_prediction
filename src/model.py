"""
model.py
--------
Neural network for binary diabetes classification.

WHY A MULTILAYER PERCEPTRON (MLP):
    EWC requires backpropagation to compute gradients — this is only
    possible with neural networks. Random Forest and XGBoost cannot be
    used with EWC because they have no differentiable weights.

    MLP is the right choice for tabular BRFSS data because:
        - Each row is an independent patient record (no sequence structure)
        - Features are numerical survey responses (no spatial structure)
        - The network is small enough to train quickly on CPU or GPU

ARCHITECTURE DECISIONS:
    128 → 64 → 32 neurons: enough capacity to hold knowledge from
    three temporal tasks simultaneously without redundant parameters.

    BatchNorm: stabilises training when data distributions shift between
    years — each BRFSS year has slightly different feature statistics.

    Dropout(0.2): light regularisation that complements the EWC penalty
    without preventing the model from learning new temporal patterns.

    Sigmoid output: produces a probability (0 to 1) for diabetes diagnosis.
"""

import torch.nn as nn


class DiabetesNet(nn.Module):
    """Feedforward neural network for diabetes binary classification."""

    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def init_model(input_size):
    """Create and return a freshly initialised DiabetesNet."""
    model = DiabetesNet(input_size)
    print(f"\nModel ready — {input_size} input features")
    print(model)
    return model
