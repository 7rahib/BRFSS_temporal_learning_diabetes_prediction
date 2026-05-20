"""
ewc.py
------
Elastic Weight Consolidation (EWC) with Fisher normalisation.

WHAT EWC DOES IN ONE SENTENCE:
    After training on each year's data, EWC identifies which neural network
    weights were most important for that year, then penalises the model for
    changing those weights when learning from the next year.

THE CORE FORMULA:
    L_EWC = L_new + λ × Σᵢ Fᵢ × (θᵢ − θ*ᵢ)²

    L_new  = classification loss on current year's data
    λ      = penalty strength (lambda)
    Fᵢ     = Fisher score for weight i (how important it was)
    θᵢ     = current weight value
    θ*ᵢ    = saved weight value from after the previous year's training

THE FISHER INFORMATION MATRIX:
    Measures how sensitive the model's output is to each weight.
    Computed as the average squared gradient over training samples.
    High Fisher = this weight matters a lot = protect it.
    Low Fisher  = this weight barely matters = allow it to change.

WHY WE NORMALISE THE FISHER MATRIX:
    Raw Fisher values are extremely small (~0.00001) because gradients are
    tiny when the model has converged. This makes the EWC penalty negligible
    regardless of lambda. Normalising to max=1.0 fixes this — lambda then
    directly and interpretably controls the protection strength.

Reference:
    Kirkpatrick et al. (2017) — "Overcoming Catastrophic Forgetting in
    Neural Networks" — Google DeepMind — PNAS
"""

import torch
import torch.nn as nn


class EWC:
    """
    Elastic Weight Consolidation.

    Call this after training on a task to snapshot the model's important
    weights. Pass the resulting EWC object into train_ewc() for the next task.

    Args:
        model       : trained model after completing a task
        dataloader  : training DataLoader for the completed task
        max_samples : how many samples to use for Fisher computation
        normalise   : whether to normalise Fisher to [0, 1] — keep True
    """

    def __init__(self, model, dataloader, max_samples=2000, normalise=True):
        self.model       = model
        self.max_samples = max_samples
        self.normalise   = normalise

        # Snapshot the current weights — these become θ* for this task
        self.saved_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        # Compute importance scores for each weight
        self.fisher = self._compute_fisher(dataloader)

        if normalise:
            self._normalise_fisher()

        self._print_stats()

    def _compute_fisher(self, dataloader):
        """
        Approximate the Fisher Information Matrix using per-sample gradients.

        For each sample: forward pass → compute loss → backward pass →
        square the gradients → accumulate. Average over all samples used.

        Per-sample computation (rather than per-batch) gives a more accurate
        estimate of each weight's true importance.
        """
        fisher       = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        criterion    = nn.BCELoss()
        sample_count = 0

        self.model.eval()

        for X_batch, y_batch in dataloader:
            if self.max_samples and sample_count >= self.max_samples:
                break

            for i in range(len(X_batch)):
                if self.max_samples and sample_count >= self.max_samples:
                    break

                self.model.zero_grad()
                out  = self.model(X_batch[i].unsqueeze(0)).reshape(-1)
                loss = criterion(out, y_batch[i].unsqueeze(0))
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.detach() ** 2

                sample_count += 1

        # Average over all samples
        for name in fisher:
            fisher[name] /= max(sample_count, 1)

        print(f"  Fisher computed over {sample_count} samples.")
        return fisher

    def _normalise_fisher(self):
        """Scale all Fisher values so the global maximum equals 1.0."""
        global_max = max(
            f.max().item() for f in self.fisher.values() if f.max().item() > 0
        )
        if global_max > 0:
            for name in self.fisher:
                self.fisher[name] /= global_max

    def _print_stats(self):
        """Print Fisher stats so we can verify normalisation worked."""
        all_vals = torch.cat([f.flatten() for f in self.fisher.values()])
        print(f"  Fisher stats (normalised={self.normalise}): "
              f"min={all_vals.min():.4f}  max={all_vals.max():.4f}  "
              f"mean={all_vals.mean():.6f}")
        print(f"  → Max should be 1.000 if normalisation is working.")

    def penalty(self, model, lambda_):
        """
        Compute the EWC regularisation penalty for the current model.

        Returns a scalar that is added to the task loss during training.
        Weights with high Fisher scores that have drifted far from their
        saved values incur a large penalty.
        """
        loss = 0.0
        for name, param in model.named_parameters():
            loss += (self.fisher[name] * (param - self.saved_weights[name]) ** 2).sum()
        return lambda_ * loss
