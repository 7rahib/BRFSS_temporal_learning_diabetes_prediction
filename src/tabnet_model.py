"""
tabnet_model.py
----------------
NEW FILE — added to compare TabNet against FT-Transformer (model.py) on
the same BRFSS temporal pipeline. Nothing in model.py, train.py, ewc.py,
replay.py, dataset.py, or evaluate.py needed to change for this — see
main.py for the only other file touched.

A simplified TabNet-style model for binary diabetes classification on
tabular BRFSS data.

Reference: Arik & Pfister (2021) — "TabNet: Attentive Interpretable
Tabular Learning" (the original TabNet paper).

WHY TABNET (alongside FT-Transformer):
    FT-Transformer looks at every feature at once with full self-attention.
    TabNet instead makes its decision in a few sequential "steps" — at
    each step it learns a soft mask that focuses on a different subset
    of features, then processes only that subset. Comparing the two
    shows whether sequential, sparse feature selection (TabNet) or full
    pairwise attention (FT-Transformer) works better for BRFSS.

    EWC still works unchanged: TabNet here is a standard nn.Module with
    differentiable weights, so the same Fisher-information penalty from
    ewc.py applies with no modifications needed there.

SIMPLIFICATIONS FROM THE ORIGINAL PAPER (kept on purpose, to keep this
model simple and easy to follow):
    - softmax is used for the attentive mask instead of sparsemax. Both
      pick which features to focus on; softmax is simpler to implement,
      at the cost of the mask not being fully sparse.
    - Plain BatchNorm1d is used instead of Ghost Batch Norm.

ARCHITECTURE (repeated for each of `n_steps` decision steps):
    1. Attentive Transformer — Linear + BatchNorm turns the previous
       step's "info" vector into a soft mask over the `input_size`
       features (which features to focus on this step). A running
       `prior` discourages re-focusing on features already used in an
       earlier step.
    2. Masked input — the mask is multiplied elementwise with the raw
       input features.
    3. Feature Transformer — Linear + BatchNorm + GLU processes the
       masked input into a hidden representation, split into:
         - "decision" half — added into the running output
         - "info" half     — fed into the next step's attentive mask
    After all steps, the summed decision output goes through a small
    linear head that outputs a single raw logit (same convention as
    FT-Transformer — see model.py: apply torch.sigmoid() to this logit
    to get a 0-1 probability, and train.py's weighted_bce_loss expects
    this raw logit directly).

CONFIGURABLE PARAMETERS (all set from main.py — no need to edit this file):
    hidden_dim — width of the feature/attentive transformers.
    n_steps    — number of sequential decision steps.
    gamma      — relaxation factor (>1) controlling how much a feature
                 already used in an earlier step can be reused later.
                 Higher = more reuse allowed.
    dropout    — dropout applied after each feature transformer.
"""

import torch
import torch.nn as nn


class GLUBlock(nn.Module):
    """Linear -> BatchNorm -> GLU (Gated Linear Unit)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.bn     = nn.BatchNorm1d(out_dim * 2)

    def forward(self, x):
        x = self.bn(self.linear(x))
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class TabNet(nn.Module):
    """Simplified TabNet: sequential attentive feature selection."""

    def __init__(self, input_size, hidden_dim=32, n_steps=3, gamma=1.5, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_steps    = n_steps
        self.gamma      = gamma

        # One feature transformer and one attentive transformer per step.
        # Each feature transformer outputs hidden_dim*2 values, split into
        # a "decision" half (size hidden_dim) and an "info" half (size hidden_dim).
        # out_dim=hidden_dim*2 because each step's output is split in half
        # below: one hidden_dim-sized "decision" chunk + one hidden_dim-sized
        # "info" chunk (see forward()).
        self.feature_transformers = nn.ModuleList([
            GLUBlock(input_size, hidden_dim * 2) for _ in range(n_steps)
        ])
        self.attentive_transformers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, input_size), nn.BatchNorm1d(input_size))
            for _ in range(n_steps)
        ])
        self.dropout = nn.Dropout(dropout)

        # Initial "info" vector that seeds the first attentive mask
        self.initial_info = nn.Parameter(torch.zeros(1, hidden_dim))

        # Classification head — outputs a single raw logit (see model.py convention)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, input_size)
        batch_size = x.size(0)

        prior = torch.ones_like(x)                        # feature-reuse prior, starts at 1 for every feature
        info  = self.initial_info.expand(batch_size, -1)   # seeds the first attentive mask
        decision_out = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for step in range(self.n_steps):
            # Attentive Transformer: decide which features to focus on this step
            mask_logits = self.attentive_transformers[step](info) * prior
            mask        = torch.softmax(mask_logits, dim=-1)
            prior       = prior * (self.gamma - mask)  # discourage reusing the same features next step

            # Feature Transformer: process only the masked features
            masked_x = mask * x
            out      = self.dropout(self.feature_transformers[step](masked_x))

            decision, info = out.chunk(2, dim=-1)
            decision_out = decision_out + torch.relu(decision)

        return self.head(decision_out)


def init_tabnet_model(input_size, hidden_dim=32, n_steps=3, gamma=1.5, dropout=0.1):
    """Create and return a freshly initialised TabNet."""
    model = TabNet(input_size, hidden_dim, n_steps, gamma, dropout)
    print(f"\nModel ready — TabNet | {input_size} input features")
    print(f"  Hidden dim: {hidden_dim} | Steps: {n_steps} | Gamma: {gamma} | Dropout: {dropout}")
    return model
