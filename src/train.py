"""
train.py
--------
Three training functions:

    train_normal()     — standard training, used for Phase A (No-EWC baseline)
    train_ewc()        — EWC-protected training, fixed LR, no scheduler
    train_ewc_replay() — EWC + Replay Buffer combined

KEY DESIGN DECISIONS:

    Fixed LR in EWC training (no scheduler):
        ReduceLROnPlateau was dropping LR to ~0.000031 by epoch 40.
        At that point weight changes are so tiny that (theta - theta*)^2
        is negligible, making the EWC penalty invisible regardless of lambda.
        Fixed LR keeps weight movement consistent, keeping EWC active.

    Replay ratio 0.25:
        25% of every batch is drawn from the replay buffer.
        Previous version used 500 total samples — too few against 300k
        training samples. At 2000 samples per task with 25% batch mixing,
        the replay signal is strong enough to meaningfully reinforce past tasks.

    Gradient clipping (max_norm=1.0):
        Prevents exploding gradients when EWC penalty and task loss
        pull in opposite directions during EWC+Replay training.
"""

import torch
import torch.nn as nn


def train_normal(model, dataloader, epochs=50, lr=0.001):
    """
    Standard training — no EWC, no replay. Uses LR scheduler.
    Used for Phase A (No-EWC sequential baseline).

    Returns: (model, loss_history)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()
    history   = []

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(X).reshape(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(dataloader)
        scheduler.step(avg)
        history.append({'epoch': epoch+1, 'total_loss': avg,
                        'task_loss': avg, 'ewc_loss': 0.0})

        
        print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model, history


def train_ewc(model, dataloader, ewc_objects, lambda_, epochs=50, lr=0.001):
    """
    EWC-protected training. Fixed LR — no scheduler.

    The EWC penalty is accumulated across ALL previous tasks.
    Fixed LR ensures weight changes remain meaningful throughout
    training, keeping the EWC penalty active until the final epoch.

    Returns: (model, loss_history)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    history   = []

    model.train()
    for epoch in range(epochs):
        total = task_sum = ewc_sum = 0.0

        for X, y in dataloader:
            optimizer.zero_grad()
            task_loss = criterion(model(X).reshape(-1), y)
            ewc_loss  = sum(e.penalty(model, lambda_) for e in ewc_objects)
            loss      = task_loss + ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total    += loss.item()
            task_sum += task_loss.item()
            ewc_sum  += ewc_loss.item()

        n = len(dataloader)
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
                  f"LR: {lr:.6f} (fixed)")

    return model, history


def train_ewc_replay(model, dataloader, ewc_objects, replay_buffer,
                     lambda_, epochs=50, lr=0.001):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
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
            task_loss = criterion(model(X).reshape(-1), y)
            ewc_loss  = sum(e.penalty(model, lambda_) for e in ewc_objects)
            loss      = task_loss + ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total    += loss.item()
            task_sum += task_loss.item()
            ewc_sum  += ewc_loss.item()

        n = len(dataloader)
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
                  f"LR: {lr:.6f} (fixed)")

    return model, history
