"""
train.py
--------
Training functions for standard training and EWC-protected training.

TWO FUNCTIONS:
    train_normal() — used for Task 1 (nothing to protect yet) and baselines
    train_ewc()    — used for Tasks 2 and 3 (EWC penalty active)

LOSS HISTORY:
    Both functions return a history list — one entry per epoch containing
    the total loss, task loss, and EWC loss. This is used to plot training
    curves and diagnose issues (e.g. if EWC loss is too small, lambda
    needs to be higher).

TRAINING CHOICES:
    Adam optimiser: adapts learning rate per parameter, converges faster
    than SGD, standard choice for tabular health data.

    ReduceLROnPlateau: halves the learning rate if loss stops improving
    for 5 epochs. Helps fine-tune near convergence.

    Half learning rate for EWC training: when EWC is active, the loss
    landscape is more constrained. Smaller steps navigate it more carefully
    and prevent the model from violating EWC constraints.

    Gradient clipping (max_norm=1.0): prevents exploding gradients when
    the EWC penalty and task loss pull weights in opposite directions.
"""

import torch
import torch.nn as nn


def train_normal(model, dataloader, epochs=50, lr=0.001):
    """
    Standard training with no EWC protection.
    Used for Task 1 (first year) and standalone baselines.

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

        history.append({
            'epoch':      epoch + 1,
            'total_loss': avg_loss,
            'task_loss':  avg_loss,
            'ewc_loss':   0.0,
        })

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

    return model, history


def train_ewc(model, dataloader, ewc_objects, lambda_, epochs=50, lr=0.001):
    """
    EWC-protected training for Tasks 2 and 3.

    The EWC penalty is summed across ALL previous tasks — so when training
    on Task 3, the model is penalised for changing weights that were
    important for both Task 1 (BRFSS 2015) and Task 2 (BRFSS 2019).

    Args:
        model       : current model to train
        dataloader  : DataLoader for the current year's data
        ewc_objects : list of EWC objects from all previous years
        lambda_     : penalty strength
        epochs      : training epochs
        lr          : base learning rate (halved internally for EWC)

    Returns: (trained model, loss history)
    """
    # Use half the learning rate — EWC constrains the loss landscape
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()
    history   = []

    model.train()

    for epoch in range(epochs):
        total_loss = task_total = ewc_total = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            output    = model(X_batch).reshape(-1)
            task_loss = criterion(output, y_batch)

            # Sum EWC penalties from all previous tasks
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
        scheduler.step(avg_total)

        history.append({
            'epoch':      epoch + 1,
            'total_loss': avg_total,
            'task_loss':  task_total / n,
            'ewc_loss':   ewc_total  / n,
        })

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} | "
                  f"Total: {avg_total:.4f} | "
                  f"Task: {task_total/n:.4f} | "
                  f"EWC: {ewc_total/n:.4f} | "
                  f"LR: {lr_now:.6f}")

    return model, history
