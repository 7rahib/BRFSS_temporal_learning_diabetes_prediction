"""
ewc.py
------
Elastic Weight Consolidation (EWC) with Fisher normalisation.

FORMULA:
    L_EWC = L_new + λ × Σᵢ Fᵢ × (θᵢ − θ*ᵢ)²

    L_new  = task loss for current year
    λ      = penalty strength (lambda)
    Fᵢ     = Fisher score (importance of weight i for previous year)
    θ*ᵢ    = saved weight value after previous year's training

FISHER NORMALISATION:
    Raw Fisher values are ~0.00001 after convergence. Normalising to
    max=1.0 makes lambda directly and interpretably control protection.

Reference: Kirkpatrick et al. (2017) — Google DeepMind — PNAS
"""

import torch
import torch.nn as nn


class EWC:
    """
    Compute and store EWC state after training on one task.
    Pass to train_ewc() as part of the ewc_objects list.
    """

    def __init__(self, model, dataloader, max_samples=2000, normalise=True):
        self.max_samples = max_samples
        self.normalise   = normalise

        # Snapshot current weights — θ* for this task
        self.saved_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        self.fisher = self._compute_fisher(model, dataloader)

        if normalise:
            self._normalise_fisher()

        self._print_stats()

    def _compute_fisher(self, model, dataloader):
        fisher    = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        criterion = nn.BCELoss()
        count     = 0

        model.eval()
        for X_batch, y_batch in dataloader:
            if self.max_samples and count >= self.max_samples:
                break
            for i in range(len(X_batch)):
                if self.max_samples and count >= self.max_samples:
                    break
                model.zero_grad()
                out  = model(X_batch[i].unsqueeze(0)).reshape(-1)
                loss = criterion(out, y_batch[i].unsqueeze(0))
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.detach() ** 2
                count += 1

        for name in fisher:
            fisher[name] /= max(count, 1)

        print(f"  Fisher computed over {count} samples.")
        return fisher

    def _normalise_fisher(self):
        global_max = max(
            f.max().item() for f in self.fisher.values() if f.max().item() > 0
        )
        if global_max > 0:
            for name in self.fisher:
                self.fisher[name] /= global_max

    def _print_stats(self):
        all_vals = torch.cat([f.flatten() for f in self.fisher.values()])
        print(f"  Fisher: min={all_vals.min():.4f}  max={all_vals.max():.4f}  "
              f"mean={all_vals.mean():.6f}  (max should be 1.0)")

    def penalty(self, model, lambda_):
        """EWC regularisation penalty for the current model state."""
        loss = 0.0
        for name, param in model.named_parameters():
            loss += (self.fisher[name] * (param - self.saved_weights[name]) ** 2).sum()
        return lambda_ * loss
