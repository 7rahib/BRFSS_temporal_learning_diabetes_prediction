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
    Raw Fisher values are extremely skewed — a handful of parameters
    dominate, so most Fisher values sit near 0 regardless of scale.
    Normalising by dividing every value by the GLOBAL MEAN (not the max)
    rescales so the average parameter's Fisher is ~1.0, without crushing
    the rest of the network down near zero the way max-normalisation
    does. This is what lets LAMBDA_EWC actually control protection
    strength network-wide, rather than only affecting the single most
    important parameter.

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
        # Unweighted — Fisher information reflects the model's own
        # prediction sensitivity, not class balance, so no pos_weight here
        # (pos_weight is only used for the *task* loss in train.py).
        criterion = nn.BCEWithLogitsLoss()
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
        # Normalise by the MEAN Fisher value across the whole network, not
        # the single global max. Fisher information is naturally very
        # skewed (a handful of parameters dominate) — dividing by the max
        # crushes almost every other parameter's Fisher down near zero,
        # which makes LAMBDA_EWC have almost no effect on ~99% of the
        # network no matter how large it gets. Dividing by the mean keeps
        # relative importance between parameters intact, but rescales so
        # the *average* parameter sits at 1 instead of the single largest
        # one — lambda then has a real, roughly predictable effect.
        all_vals    = torch.cat([f.flatten() for f in self.fisher.values()])
        global_mean = all_vals.mean().item()
        if global_mean > 0:
            for name in self.fisher:
                self.fisher[name] /= global_mean

    def _print_stats(self):
        all_vals = torch.cat([f.flatten() for f in self.fisher.values()])
        print(f"  Fisher: min={all_vals.min():.4f}  max={all_vals.max():.4f}  "
              f"mean={all_vals.mean():.6f}  (mean should be ~1.0)")

    def penalty(self, model, lambda_):
        """EWC regularisation penalty for the current model state."""
        loss = 0.0
        for name, param in model.named_parameters():
            loss += (self.fisher[name] * (param - self.saved_weights[name]) ** 2).sum()
        return lambda_ * loss
