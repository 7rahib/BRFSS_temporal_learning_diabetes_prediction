"""
train.py
--------
Three training functions:

    train_normal()     — standard training, used for Phase A (No-EWC baseline)
    train_ewc()        — EWC-protected training
    train_ewc_replay() — EWC + Replay Buffer combined

KEY DESIGN DECISIONS:

    Weighted BCELoss (class imbalance):
        The model outputs a raw logit (see model.py), so a sigmoid is
        applied here to get a probability, then nn.BCELoss is used with a
        per-sample `weight` — positive (diabetic) samples are weighted by
        `pos_weight` (see dataset.compute_pos_weight()), negative samples
        stay at 1.0. This directly addresses the ~83-84% "predict majority
        class" plateau seen without it.

    Warmup + cosine-decay LR scheduler (with a floor):
        Attention-based models like FT-Transformer tend to train more
        stably with a short LR warmup rather than the full LR from step 1,
        and benefit from decaying afterward instead of staying flat.
        BUT: EWC's penalty is lambda * Fisher * (theta - theta*)^2 — if LR
        decays too close to zero, weight movement becomes negligible and
        the penalty stops doing anything regardless of lambda (this is
        why the original version used a fixed LR for EWC training). The
        scheduler below solves both: it warms up, then decays with cosine
        annealing down to a floor (`min_lr_ratio` of the peak LR) instead
        of all the way to zero, so weight movement — and therefore EWC —
        stays meaningful for the entire run.

    AdamW (not Adam) with weight_decay:
        Plain Adam's `weight_decay` argument applies L2 regularisation
        added directly into the gradient, which interacts oddly with
        Adam's adaptive per-parameter learning rates. AdamW decouples
        weight decay from the gradient update instead, which is the
        standard choice for transformer-style models — same interface,
        more correct behaviour, so it's a drop-in replacement here.

    Replay ratio 0.25:
        25% of every batch is drawn from the replay buffer.
        Previous version used 500 total samples — too few against 300k
        training samples. At 2000 samples per task with 25% batch mixing,
        the replay signal is strong enough to meaningfully reinforce past tasks.

    Gradient clipping (max_norm=1.0):
        Prevents exploding gradients when EWC penalty and task loss
        pull in opposite directions during EWC+Replay training.
"""

import math
import torch
import torch.nn as nn


def weighted_bce_loss(logits, y, pos_weight=None):
    """
    Weighted binary cross-entropy.

    Applies sigmoid to the model's raw logit to get a probability, then
    computes BCELoss with a per-sample weight: positive (diabetic)
    samples are weighted by `pos_weight`, negative samples stay at 1.0.
    This is what corrects for class imbalance (see dataset.compute_pos_weight).

    pos_weight=None -> plain, unweighted BCELoss.
    """
    probs = torch.sigmoid(logits)
    if pos_weight is None:
        return nn.functional.binary_cross_entropy(probs, y)
    weight = torch.where(y == 1, pos_weight, torch.ones_like(y))
    return nn.functional.binary_cross_entropy(probs, y, weight=weight)


def build_scheduler(optimizer, total_epochs, warmup_epochs=None, min_lr_ratio=0.3):
    """
    Linear warmup for `warmup_epochs`, then cosine decay down to
    `min_lr_ratio` x peak LR for the rest of training.

    warmup_epochs=None -> defaults to 10% of total_epochs (at least 1).
    """
    if warmup_epochs is None:
        warmup_epochs = max(1, total_epochs // 10)
    warmup_epochs = min(warmup_epochs, max(1, total_epochs - 1))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        span     = max(1, total_epochs - warmup_epochs)
        progress = min((epoch - warmup_epochs) / span, 1.0)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_normal(model, dataloader, epochs=50, lr=0.001, pos_weight=None,
                  warmup_epochs=None, min_lr_ratio=0.3):
    """
    Standard training — no EWC, no replay.
    Used for Phase A (No-EWC sequential baseline).

    Returns: (model, loss_history)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio)
    history   = []

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            loss = weighted_bce_loss(model(X).reshape(-1), y, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(dataloader)
        scheduler.step()
        history.append({'epoch': epoch+1, 'total_loss': avg,
                        'task_loss': avg, 'ewc_loss': 0.0})

        print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model, history


def train_ewc(model, dataloader, ewc_objects, lambda_, epochs=50, lr=0.001,
              pos_weight=None, warmup_epochs=None, min_lr_ratio=0.3):
    """
    EWC-protected training.

    The EWC penalty is accumulated across ALL previous tasks. The LR
    scheduler decays to a floor (min_lr_ratio) rather than to zero, so
    weight changes stay large enough for the EWC penalty to remain active
    for the whole run (see module docstring).

    Returns: (model, loss_history)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio)
    history   = []

    model.train()
    for epoch in range(epochs):
        total = task_sum = ewc_sum = 0.0

        for X, y in dataloader:
            optimizer.zero_grad()
            task_loss = weighted_bce_loss(model(X).reshape(-1), y, pos_weight)
            ewc_loss  = sum(e.penalty(model, lambda_) for e in ewc_objects)
            loss      = task_loss + ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total    += loss.item()
            task_sum += task_loss.item()
            ewc_sum  += ewc_loss.item()

        n = len(dataloader)
        scheduler.step()
        history.append({
            'epoch':      epoch + 1,
            'total_loss': total    / n,
            'task_loss':  task_sum / n,
            'ewc_loss':   ewc_sum  / n,
        })

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | "
                  f"Total: {total/n:.4f} | "
                  f"Task: {task_sum/n:.4f} | "
                  f"EWC: {ewc_sum/n:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model, history


def train_ewc_replay(model, dataloader, ewc_objects, replay_buffer,
                     lambda_, epochs=50, lr=0.001, pos_weight=None,
                     warmup_epochs=None, min_lr_ratio=0.3):
    """
    EWC + Replay Buffer combined training.

    Each batch is augmented with replay samples drawn from all past tasks.
    The combined batch is then trained with the EWC penalty active.

    How the batch is constructed:
        - Draw the normal batch from the current task DataLoader
        - Draw replay_batch_size() samples from the replay buffer
        - Concatenate both — model sees new and old data every step

    This means the model is simultaneously:
        1. Learning new patterns (current task data)
        2. Reminded of old patterns (replay samples)
        3. Penalised for forgetting important weights (EWC)

    Returns: (model, loss_history)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio)
    history   = []

    model.train()
    for epoch in range(epochs):
        total = task_sum = ewc_sum = 0.0

        for X, y in dataloader:

            # Augment batch with replay samples if buffer is not empty
            if not replay_buffer.is_empty:
                n_replay        = replay_buffer.replay_batch_size(len(X))
                X_rep, y_rep    = replay_buffer.get_replay_batch(n_replay)
                X = torch.cat([X, X_rep], dim=0)
                y = torch.cat([y, y_rep], dim=0)

            optimizer.zero_grad()
            task_loss = weighted_bce_loss(model(X).reshape(-1), y, pos_weight)
            ewc_loss  = sum(e.penalty(model, lambda_) for e in ewc_objects)
            loss      = task_loss + ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total    += loss.item()
            task_sum += task_loss.item()
            ewc_sum  += ewc_loss.item()

        n = len(dataloader)
        scheduler.step()
        history.append({
            'epoch':      epoch + 1,
            'total_loss': total    / n,
            'task_loss':  task_sum / n,
            'ewc_loss':   ewc_sum  / n,
        })

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | "
                  f"Total: {total/n:.4f} | "
                  f"Task: {task_sum/n:.4f} | "
                  f"EWC: {ewc_sum/n:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model, history
