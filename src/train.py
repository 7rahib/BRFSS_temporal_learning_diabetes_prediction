"""
train.py
--------
Training functions for standard training and EWC-protected training.

THREE FUNCTIONS:
    train_normal()      — Task 1 and standalone baselines. Uses a learning
                          rate scheduler because there are no EWC constraints.

    train_ewc()         — Tasks 2 and 3 (EWC only). Uses a FIXED learning
                          rate with NO scheduler. This is the critical fix.

    train_ewc_replay()  — Tasks 2 and 3 (EWC + Replay). Trains on a
                          combined DataLoader that mixes current task data
                          with replayed past samples. Then adds EWC penalty
                          on top. The two mechanisms are complementary:
                          replay fights forgetting empirically (by showing old
                          data again), EWC fights it mathematically (by
                          penalising weight drift on important parameters).

WHY NO SCHEDULER IN EWC TRAINING:
    The ReduceLROnPlateau scheduler drops the learning rate when loss
    plateaus. In EWC training, the total loss plateaus quickly because
    the EWC penalty resists weight changes. The scheduler then drops LR
    to very small values (~0.000031) which means weights barely move at all.
    When weights barely move, (theta - theta*)^2 is tiny, so the EWC
    penalty becomes negligible regardless of lambda.

    With a fixed LR, weights keep moving at a consistent rate throughout
    training, which keeps the EWC penalty active and meaningful.

GRADIENT CLIPPING:
    max_norm=1.0 prevents exploding gradients when the EWC penalty and
    task loss pull weights in opposite directions. Essential for stability.
"""

import torch
import torch.nn as nn


def train_normal(model, dataloader, epochs=50, lr=0.001):
    """
    Standard training — no EWC. Uses LR scheduler for convergence.
    Returns: (trained model, loss history)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()
    history   = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch).reshape(-1)
            loss   = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        history.append({'epoch': epoch+1, 'total_loss': avg_loss,
                        'task_loss': avg_loss, 'ewc_loss': 0.0})

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

    return model, history


def train_ewc(model, dataloader, ewc_objects, lambda_, epochs=50, lr=0.001):
    """
    EWC-protected training for Tasks 2 and 3.

    KEY DIFFERENCE FROM train_normal:
        No learning rate scheduler. Fixed LR throughout.
        This keeps weight updates consistent, which keeps the EWC
        penalty (theta - theta*)^2 meaningful across all epochs.

    The EWC penalty is summed across ALL previous tasks — so when
    training on BRFSS 2023, the model is penalised for changing weights
    that were important for both 2015 and 2019.

    Args:
        model       : current model to train
        dataloader  : DataLoader for current year
        ewc_objects : list of EWC objects from all previous years
        lambda_     : EWC penalty strength
        epochs      : training epochs
        lr          : fixed learning rate (not halved — scheduler removed)

    Returns: (trained model, loss history)
    """
    # Fixed LR — no scheduler. Essential for keeping EWC penalty active.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    history   = []

    model.train()
    for epoch in range(epochs):
        total_loss = task_total = ewc_total = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            output    = model(X_batch).reshape(-1)
            task_loss = criterion(output, y_batch)

            # Sum EWC penalties from ALL previous years
            ewc_loss = sum(ewc.penalty(model, lambda_) for ewc in ewc_objects)

            loss = task_loss + ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            task_total += task_loss.item()
            ewc_total  += ewc_loss.item()

        n         = len(dataloader)
        avg_total = total_loss / n
        history.append({
            'epoch':      epoch + 1,
            'total_loss': avg_total,
            'task_loss':  task_total / n,
            'ewc_loss':   ewc_total  / n,
        })

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | "
                  f"Total: {avg_total:.4f} | "
                  f"Task: {task_total/n:.4f} | "
                  f"EWC: {ewc_total/n:.4f} | "
                  f"LR: {lr:.6f} (fixed)")

    return model, history


def train_ewc_replay(model, combined_loader, ewc_objects, lambda_, epochs=50, lr=0.001):
    """
    EWC + Replay training for Tasks 2 and 3.

    HOW IT DIFFERS FROM train_ewc():
        The dataloader passed in is a COMBINED loader (current task data +
        replayed past samples). This is built by ReplayBuffer.get_combined_loader()
        before calling this function.

        Everything else — fixed LR, EWC penalty, gradient clipping — is identical
        to train_ewc(). The two mechanisms simply stack on top of each other:

            Total Loss = BCE(current + replay data) + λ × EWC_penalty

        Replay ensures the model sees old examples again every epoch.
        EWC ensures the model's important weights don't drift, even on examples
        not covered by the buffer.

    WHY FIXED LR (same reason as train_ewc):
        Scheduler would drop LR and make EWC penalty negligible.
        Fixed LR keeps weight movement — and therefore EWC — active.

    Args:
        model           : current model to train
        combined_loader : DataLoader mixing current + replayed past data
                          (built by ReplayBuffer.get_combined_loader)
        ewc_objects     : list of EWC objects from all previous years
        lambda_         : EWC penalty strength (same value as train_ewc)
        epochs          : training epochs
        lr              : fixed learning rate

    Returns: (trained model, loss history)
    """
    # Fixed LR — same as train_ewc, no scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    history   = []

    model.train()
    for epoch in range(epochs):
        total_loss = task_total = ewc_total = 0.0

        # Each batch may contain a mix of current-year + replayed examples.
        # The BCE loss treats them equally — the model must be correct on all.
        for X_batch, y_batch in combined_loader:
            optimizer.zero_grad()

            output    = model(X_batch).reshape(-1)
            task_loss = criterion(output, y_batch)

            # EWC penalty on top of the replay-augmented task loss
            ewc_loss = sum(ewc.penalty(model, lambda_) for ewc in ewc_objects)

            loss = task_loss + ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            task_total += task_loss.item()
            ewc_total  += ewc_loss.item()

        n         = len(combined_loader)
        avg_total = total_loss / n
        history.append({
            'epoch':      epoch + 1,
            'total_loss': avg_total,
            'task_loss':  task_total / n,
            'ewc_loss':   ewc_total  / n,
        })

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | "
                  f"Total: {avg_total:.4f} | "
                  f"Task (+ replay): {task_total/n:.4f} | "
                  f"EWC: {ewc_total/n:.4f} | "
                  f"LR: {lr:.6f} (fixed)")

    return model, history
