"""
evaluate.py
-----------
Evaluation metrics and graphs for temporal incremental learning.

METRICS USED:
    Accuracy   — % correct predictions. Misleading with class imbalance
                 (a model predicting everyone as non-diabetic gets ~83%
                 accuracy on BRFSS but catches zero diabetic patients).

    Recall     — % of actual diabetics correctly identified. THE most
                 important clinical metric. Low recall = missed diagnoses.

    Precision  — % of predicted diabetics who actually have diabetes.

    F1 Score   — harmonic mean of precision and recall. Best single metric
                 for imbalanced datasets. Punishes models that sacrifice
                 one for the other.

    AUC-ROC    — area under the ROC curve. Threshold-independent measure
                 of discriminative ability. Comparable between EWC and
                 No-EWC even when accuracy differs significantly.

CONTINUAL LEARNING METRICS:
    Backward Transfer (BWT) — how much did training on new years affect
        performance on older years? Closer to 0% = better retention.
        Negative = forgetting. Positive = improvement (but check recall
        — accuracy gains can be fake due to threshold shift).

    Forward Transfer (FWT) — did learning earlier years give the model
        a head start on later years before they were trained?
        Positive = beneficial knowledge transfer forward.

GRAPHS PRODUCED:
    01 — Final accuracy comparison: No-EWC vs EWC (bar chart)
    02 — EWC accuracy over training stages (line chart)
    03 — No-EWC forgetting in action (line chart)
    04 — Backward transfer per year (bar chart)
    05 — Forward transfer per year (bar chart)
    06 — Transfer summary (BWT + FWT combined)
    07 — Confusion matrices: No-EWC vs EWC side by side
    08 — ROC curves: No-EWC vs EWC
    09 — Precision / Recall / F1 comparison
    10a — Forgetting heatmap with EWC
    10b — Forgetting heatmap without EWC
    11 — Training loss curves (task loss vs EWC penalty)
    12 — EWC penalty / task loss ratio (diagnoses lambda issues)
    13 — Calibrated metrics (per-task optimal threshold)
    14 — Optimal threshold per task
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)

os.makedirs('results', exist_ok=True)


# ──────────────────────────────────────────
# CORE PREDICTION
# ──────────────────────────────────────────

def get_predictions(model, dataloader):
    """Run the model on a DataLoader. Returns (labels, predictions, probabilities)."""
    model.eval()
    labels_all = []
    preds_all  = []
    probs_all  = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            out   = model(X_batch).reshape(-1)
            preds = (out >= 0.5).float()
            labels_all.extend(y_batch.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
            probs_all.extend(out.cpu().numpy())

    return np.array(labels_all), np.array(preds_all), np.array(probs_all)


def evaluate(model, dataloader):
    """Return accuracy (0.0 to 1.0)."""
    labels, preds, _ = get_predictions(model, dataloader)
    return accuracy_score(labels, preds)


def full_metrics(model, dataloader, task_name):
    """Compute and print all metrics for one task. Returns a dict."""
    labels, preds, probs = get_predictions(model, dataloader)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')

    print(f"\n  [{task_name}]")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%  ← most important clinically")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    AUC-ROC   : {auc:.4f}")

    return {
        'Task':      task_name,
        'Accuracy':  round(acc  * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall':    round(rec  * 100, 2),
        'F1':        round(f1   * 100, 2),
        'AUC':       round(float(auc), 4),
    }


def evaluate_all_tasks(model, test_loaders, task_names, current_task_idx):
    results = {}

    for j in range(current_task_idx + 1):
        acc = evaluate(model, test_loaders[j])

        results[task_names[j]] = round(acc * 100, 2)

        print(f"    {task_names[j]}: {acc * 100:.2f}%")

    return results


# ──────────────────────────────────────────
# CONTINUAL LEARNING METRICS
# ──────────────────────────────────────────

def compute_backward_transfer(results_log, task_names):
    """
    Backward Transfer (BWT): how much did later tasks affect earlier ones?

    BWT = average [ R(final, i) - R(i, i) ] for all tasks except the last

    Where R(stage, task) is accuracy on 'task' after training stage 'stage'.
    R(i, i) = accuracy on task i right after it was trained (best case).
    R(final, i) = accuracy on task i after all tasks have been trained.

    BWT = 0   → perfect retention
    BWT < 0   → forgetting occurred
    BWT > 0   → later tasks improved earlier task accuracy (check recall!)
    """
    final        = results_log[-1]
    per_task_bwt = {}

    for i, name in enumerate(task_names[:-1]):
        acc_at_training = results_log[i].get(name)
        acc_final       = final.get(name)
        if acc_at_training is not None and acc_final is not None:
            per_task_bwt[name] = round(acc_final - acc_at_training, 2)

    overall = round(np.mean(list(per_task_bwt.values())), 2) if per_task_bwt else 0.0

    print(f"\n  Backward Transfer (BWT): {overall:+.2f}%")
    for name, val in per_task_bwt.items():
        status = '✓ retained' if val >= -2 else '✗ forgotten'
        print(f"    {name}: {val:+.2f}%  {status}")

    return overall, per_task_bwt


# def compute_forward_transfer(results_log, task_names, baseline_accs):
#     """
#     Forward Transfer (FWT): did learning earlier years help with later years?

#     FWT = average [ R(i-1, i) - baseline_i ] for tasks i > 1

#     R(i-1, i)  = accuracy on task i measured before task i was trained
#     baseline_i = standalone accuracy (Phase A) — the reference point

#     Positive FWT → knowledge from earlier years transferred forward.
#     Negative FWT → earlier years interfered with later year learning.
#     """
#     per_task_fwt = {}

#     for i in range(1, len(task_names)):
#         name               = task_names[i]
#         acc_before_trained = results_log[i - 1].get(name)
#         baseline           = baseline_accs.get(name)

#         if acc_before_trained is not None and baseline is not None:
#             per_task_fwt[name] = round(acc_before_trained - baseline, 2)

#     overall = round(np.mean(list(per_task_fwt.values())), 2) if per_task_fwt else 0.0

#     print(f"\n  Forward Transfer (FWT): {overall:+.2f}%")
#     for name, val in per_task_fwt.items():
#         status = '✓ positive' if val >= 0 else '✗ negative'
#         print(f"    {name}: {val:+.2f}%  {status}")

#     return overall, per_task_fwt


# ──────────────────────────────────────────
# THRESHOLD CALIBRATION
# ──────────────────────────────────────────

def find_best_threshold(model, dataloader):
    """
    Find the classification threshold that maximises F1 for this task.

    Instead of always using 0.5, we search 0.05 to 0.90 and pick the
    threshold that gives the best F1 score. This corrects for class
    imbalance — BRFSS 2015 may need a lower threshold than 2023 because
    it has fewer diabetic patients in the test set.
    """
    labels, _, probs = get_predictions(model, dataloader)
    best_thresh, best_f1 = 0.5, 0.0

    for thresh in np.arange(0.05, 0.90, 0.05):
        preds = (probs >= thresh).astype(float)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh

    return round(float(best_thresh), 2), round(best_f1 * 100, 2)


def calibrate_all_tasks(model, test_loaders, task_names):
    """Find the best threshold for every task. Returns {task_name: threshold}."""
    print("\n  Per-task threshold calibration:")
    thresholds = {}
    for name, loader in zip(task_names, test_loaders):
        thresh, best_f1 = find_best_threshold(model, loader)
        thresholds[name] = thresh
        print(f"    {name}: threshold={thresh:.2f}  (best F1={best_f1:.1f}%)")
    return thresholds


def full_metrics_calibrated(model, dataloader, task_name, threshold):
    """Compute all metrics using a task-specific threshold instead of 0.5."""
    labels, _, probs = get_predictions(model, dataloader)
    preds = (probs >= threshold).astype(float)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')

    print(f"\n  [{task_name}] (threshold={threshold:.2f})")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%  <- most important clinically")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    AUC-ROC   : {auc:.4f}")

    return {
        'Task':      task_name,
        'Threshold': threshold,
        'Accuracy':  round(acc  * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall':    round(rec  * 100, 2),
        'F1':        round(f1   * 100, 2),
        'AUC':       round(float(auc), 4),
    }


# ──────────────────────────────────────────
# SAVE RESULTS
# ──────────────────────────────────────────

def save_results(results_log, task_names, filename='results.csv', output_dir='results'):
    """Save accuracy results table to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    stages = [f"After Task {i+1}" for i in range(len(results_log))]
    rows   = [{'Stage': s, **r} for s, r in zip(stages, results_log)]
    df     = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/{filename}", index=False)
    print(f"  Saved: {output_dir}/{filename}")
    return df


def save_full_metrics(metrics_list, filename, output_dir='results'):
    """Save full metrics to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(metrics_list).to_csv(f"{output_dir}/{filename}", index=False)
    print(f"  Saved: {output_dir}/{filename}")


# ──────────────────────────────────────────
# GRAPHS
# ──────────────────────────────────────────

def _save(filename, output_dir='results'):
    """Save and close the current figure."""
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_noewc_vs_ewc(noewc_final, ewc_final, task_names, output_dir='results'):
    """Bar chart: final accuracy after all tasks — No-EWC vs EWC."""
    noewc_vals = [noewc_final.get(n, 0) for n in task_names]
    ewc_vals   = [ewc_final.get(n, 0)   for n in task_names]
    x          = np.arange(len(task_names))
    w          = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, noewc_vals, w, label='Without EWC', color='#EF5350', alpha=0.85)
    b2 = ax.bar(x + w/2, ewc_vals,   w, label='With EWC',    color='#42A5F5', alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Final Accuracy After All 3 Temporal Tasks\n(Without EWC vs With EWC)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Task (Year)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('—')[-1].strip() for n in task_names])
    ax.set_ylim(40, 115)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save('01_noewc_vs_ewc.png', output_dir)


def plot_ewc_accuracy_over_stages(results_log, task_names, output_dir='results'):
    """Line chart: EWC accuracy on each task at each training stage."""
    stages = [f"After\n{n.split('—')[-1].strip()}" for n in task_names]
    colors = ['#2196F3', '#4CAF50', '#FF5722']

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(task_names):
        vals = [r[name] for r in results_log if name in r]
        xpts = stages[:len(vals)]
        plt.plot(xpts, vals, marker='o', linewidth=2.5, markersize=9,
                 label=name, color=colors[i % len(colors)])
        for x, y in zip(xpts, vals):
            plt.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                         xytext=(0, 10), ha='center', fontsize=9)

    plt.title('EWC — Accuracy on Each Year Over Training Stages', fontsize=14, fontweight='bold')
    plt.xlabel('Training Stage')
    plt.ylabel('Accuracy (%)')
    plt.ylim(50, 110)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save('02_ewc_accuracy_stages.png', output_dir)


def plot_noewc_accuracy_over_stages(results_log, task_names, output_dir='results'):
    """Line chart: No-EWC accuracy — shows catastrophic forgetting."""
    stages = [f"After\n{n.split('—')[-1].strip()}" for n in task_names]
    colors = ['#EF5350', '#FF7043', '#FFCA28']

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(task_names):
        vals = [r[name] for r in results_log if name in r]
        xpts = stages[:len(vals)]
        plt.plot(xpts, vals, marker='s', linewidth=2.5, markersize=9,
                 label=name, color=colors[i % len(colors)], linestyle='--')
        for x, y in zip(xpts, vals):
            plt.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                         xytext=(0, 10), ha='center', fontsize=9)

    plt.title('WITHOUT EWC — Temporal Forgetting In Action', fontsize=14, fontweight='bold')
    plt.xlabel('Training Stage')
    plt.ylabel('Accuracy (%)')
    plt.ylim(30, 115)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save('03_noewc_forgetting.png', output_dir)


def plot_backward_transfer(per_bwt_noewc, per_bwt_ewc, output_dir='results'):
    """Bar chart: backward transfer per task — No-EWC vs EWC."""
    tasks     = list(per_bwt_ewc.keys())
    noewc_vals = [per_bwt_noewc.get(t, 0) for t in tasks]
    ewc_vals   = [per_bwt_ewc.get(t, 0)   for t in tasks]
    x          = np.arange(len(tasks))
    w          = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, noewc_vals, w, label='Without EWC', color='#EF5350', alpha=0.85)
    ax.bar(x + w/2, ewc_vals,   w, label='With EWC',    color='#42A5F5', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Backward Transfer (BWT) — No-EWC vs EWC\n(0% = perfect retention | negative = forgetting)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Year / Task')
    ax.set_ylabel('BWT (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([t.split('—')[-1].strip() for t in tasks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save('04_backward_transfer.png', output_dir)


# def plot_forward_transfer(per_fwt_noewc, per_fwt_ewc, output_dir='results'):
#     """Bar chart: forward transfer per task — No-EWC vs EWC."""
#     tasks     = list(per_fwt_ewc.keys())
#     if not tasks:
#         return
#     noewc_vals = [per_fwt_noewc.get(t, 0) for t in tasks]
#     ewc_vals   = [per_fwt_ewc.get(t, 0)   for t in tasks]
#     x          = np.arange(len(tasks))
#     w          = 0.35

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.bar(x - w/2, noewc_vals, w, label='Without EWC', color='#EF5350', alpha=0.85)
#     ax.bar(x + w/2, ewc_vals,   w, label='With EWC',    color='#42A5F5', alpha=0.85)
#     ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
#     ax.set_title('Forward Transfer (FWT) — No-EWC vs EWC\n(positive = earlier years helped future years)',
#                  fontsize=13, fontweight='bold')
#     ax.set_xlabel('Year / Task')
#     ax.set_ylabel('FWT (%)')
#     ax.set_xticks(x)
#     ax.set_xticklabels([t.split('—')[-1].strip() for t in tasks])
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis='y')
#     _save('05_forward_transfer.png', output_dir)


# def plot_transfer_summary(bwt_noewc, bwt_ewc, fwt_noewc, fwt_ewc, output_dir='results'):
#     """Single chart showing overall BWT and FWT for both models."""
#     metrics   = ['Backward Transfer (BWT)', 'Forward Transfer (FWT)']
#     noewc_v   = [bwt_noewc, fwt_noewc]
#     ewc_v     = [bwt_ewc,   fwt_ewc]
#     x         = np.arange(len(metrics))
#     w         = 0.35

#     fig, ax = plt.subplots(figsize=(9, 6))
#     ax.bar(x - w/2, noewc_v, w, label='Without EWC', color='#EF5350', alpha=0.85)
#     ax.bar(x + w/2, ewc_v,   w, label='With EWC',    color='#42A5F5', alpha=0.85)
#     for i, (nv, ev) in enumerate(zip(noewc_v, ewc_v)):
#         ax.text(i - w/2, nv + (0.3 if nv >= 0 else -1.5), f'{nv:+.2f}%', ha='center', fontsize=11, fontweight='bold')
#         ax.text(i + w/2, ev + (0.3 if ev >= 0 else -1.5), f'{ev:+.2f}%', ha='center', fontsize=11, fontweight='bold')
#     ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
#     ax.set_title('Transfer Learning Summary\n(BWT: closer to 0 is better | FWT: higher is better)',
#                  fontsize=13, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics)
#     ax.set_ylabel('Score (%)')
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis='y')
#     _save('06_transfer_summary.png', output_dir)


def plot_confusion_matrices_comparison(noewc_model, ewc_model, test_loaders, task_names, output_dir='results'):
    """Two rows of confusion matrices: No-EWC (top) vs EWC (bottom)."""
    n   = len(task_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    for col, (loader, name) in enumerate(zip(test_loaders, task_names)):
        for row, (model, label) in enumerate([(noewc_model, 'No EWC'), (ewc_model, 'EWC')]):
            labels, preds, _ = get_predictions(model, loader)
            cm = confusion_matrix(labels, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row][col],
                        xticklabels=['No Diab.', 'Diab.'],
                        yticklabels=['No Diab.', 'Diab.'])
            axes[row][col].set_title(f'{label}\n{name.split("—")[-1].strip()}', fontsize=10, fontweight='bold')
            axes[row][col].set_xlabel('Predicted')
            axes[row][col].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices — No EWC (top) vs EWC (bottom)', fontsize=14, fontweight='bold')
    _save('07_confusion_matrices.png', output_dir)


def plot_roc_comparison(noewc_model, ewc_model, test_loaders, task_names, output_dir='results'):
    """ROC curves for all tasks: solid = EWC, dashed = No-EWC."""
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    plt.figure(figsize=(9, 7))

    for loader, name, color in zip(test_loaders, task_names, colors):
        short = name.split('—')[-1].strip()
        for model, style, suffix in [(noewc_model, '--', 'No EWC'), (ewc_model, '-', 'EWC')]:
            labs, _, probs = get_predictions(model, loader)
            try:
                fpr, tpr, _ = roc_curve(labs, probs)
                auc = roc_auc_score(labs, probs)
                plt.plot(fpr, tpr, linestyle=style, linewidth=2, color=color,
                         label=f'{short} {suffix} (AUC={auc:.3f})')
            except Exception:
                pass

    plt.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Random')
    plt.title('ROC Curves — No EWC (dashed) vs EWC (solid)', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    _save('08_roc_curves.png', output_dir)


def plot_metrics_comparison(noewc_metrics, ewc_metrics, output_dir='results'):
    """Grouped bar chart: Precision / Recall / F1 — No-EWC vs EWC."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, metric in zip(axes, ['Precision', 'Recall', 'F1']):
        noewc_v = [m[metric] for m in noewc_metrics]
        ewc_v   = [m[metric] for m in ewc_metrics]
        labels  = [m['Task'].split('—')[-1].strip() for m in ewc_metrics]
        x       = np.arange(len(labels))
        w       = 0.35
        ax.bar(x - w/2, noewc_v, w, label='No EWC', color='#EF5350', alpha=0.85)
        ax.bar(x + w/2, ewc_v,   w, label='EWC',    color='#42A5F5', alpha=0.85)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_ylabel('Score (%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Precision / Recall / F1 — No EWC vs EWC', fontsize=14, fontweight='bold')
    _save('09_precision_recall_f1.png', output_dir)


def plot_forgetting_heatmap(results_log, task_names, title_suffix, filename, output_dir='results'):
    """Colour-coded table: rows = stages, columns = tasks. Green = high accuracy."""
    stages = [f"After Task {i+1}" for i in range(len(results_log))]
    data   = [[r.get(n, float('nan')) for n in task_names] for r in results_log]
    df     = pd.DataFrame(data, index=stages, columns=task_names)

    plt.figure(figsize=(10, 5))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlGn',
                mask=df.isna(), linewidths=0.5, linecolor='grey',
                vmin=50, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
    plt.title(f'Temporal Forgetting Heatmap — {title_suffix}', fontsize=13, fontweight='bold')
    _save(filename, output_dir)


def plot_training_loss_curves(histories, task_names, output_dir='results'):
    """Per-task training loss: total, task loss, and EWC penalty."""
    fig, axes = plt.subplots(len(histories), 1, figsize=(11, 4 * len(histories)))
    if len(histories) == 1:
        axes = [axes]

    for ax, history, name in zip(axes, histories, task_names):
        epochs     = [h['epoch']      for h in history]
        total_loss = [h['total_loss'] for h in history]
        task_loss  = [h['task_loss']  for h in history]
        ewc_loss   = [h['ewc_loss']   for h in history]

        ax.plot(epochs, total_loss, 'k-',  linewidth=2,   label='Total Loss')
        ax.plot(epochs, task_loss,  'b--', linewidth=1.8, label='Task Loss')
        ax.plot(epochs, ewc_loss,   'r:',  linewidth=1.8, label='EWC Penalty')
        ax.set_title(f'Training Loss — {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if history and max(h['ewc_loss'] for h in history) > max(h['task_loss'] for h in history) * 2:
            ax.text(0.5, 0.85, 'EWC penalty dominating — consider reducing lambda',
                    transform=ax.transAxes, ha='center', fontsize=10, color='red',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Training Loss Curves — Task Loss vs EWC Penalty', fontsize=14, fontweight='bold')
    _save('11_training_loss_curves.png', output_dir)


def plot_ewc_penalty_magnitude(histories, task_names, output_dir='results'):
    """
    EWC penalty / task loss ratio over training.

    This directly answers: is EWC actually doing anything?
    Ratio between 0.5 and 3.0 = EWC is genuinely competing with task loss.
    Ratio < 0.1 = EWC is negligible — increase lambda or check normalisation.
    Ratio > 5.0 = EWC is dominating — model cannot learn new task, reduce lambda.
    """
    colors = ['#4CAF50', '#2196F3', '#FF5722']
    fig, ax = plt.subplots(figsize=(11, 6))

    for history, name, color in zip(histories, task_names, colors):
        if all(h['ewc_loss'] == 0 for h in history):
            continue
        epochs = [h['epoch'] for h in history]
        ratio  = [h['ewc_loss'] / (h['task_loss'] + 1e-8) for h in history]
        ax.plot(epochs, ratio, linewidth=2.5, label=name, color=color)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='Ratio = 1.0 (balanced)')
    ax.axhline(3.0, color='red',   linestyle=':',  linewidth=1, label='Ratio = 3.0 (EWC dominating)')
    ax.set_title('EWC Penalty / Task Loss Ratio\n(0.5–3.0 = balanced | < 0.1 = EWC too weak | > 5.0 = EWC too strong)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save('12_ewc_penalty_ratio.png', output_dir)


def plot_calibrated_metrics(noewc_cal, ewc_cal, output_dir='results'):
    """Bar chart: calibrated Precision / Recall / F1 — No-EWC vs EWC."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    thresholds = [m['Threshold'] for m in ewc_cal]

    for ax, metric in zip(axes, ['Precision', 'Recall', 'F1']):
        noewc_v = [m[metric] for m in noewc_cal]
        ewc_v   = [m[metric] for m in ewc_cal]
        labels  = [f"{m['Task'].split('—')[-1].strip()}\n(t={thresholds[i]:.2f})"
                   for i, m in enumerate(ewc_cal)]
        x       = np.arange(len(labels))
        w       = 0.35
        ax.bar(x - w/2, noewc_v, w, label='No EWC (calibrated)', color='#EF5350', alpha=0.85)
        ax.bar(x + w/2, ewc_v,   w, label='EWC (calibrated)',    color='#42A5F5', alpha=0.85)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel('Score (%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Calibrated Metrics (per-task optimal threshold) — No EWC vs EWC',
                 fontsize=13, fontweight='bold')
    _save('13_calibrated_metrics.png', output_dir)


def plot_threshold_comparison(noewc_thresholds, ewc_thresholds, task_names, output_dir='results'):
    """Bar chart: optimal threshold per task for both models."""
    noewc_v = [noewc_thresholds[n] for n in task_names]
    ewc_v   = [ewc_thresholds[n]   for n in task_names]
    x       = np.arange(len(task_names))
    w       = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, noewc_v, w, label='No EWC', color='#EF5350', alpha=0.85)
    ax.bar(x,     ewc_v,   w, label='EWC',    color='#42A5F5', alpha=0.85)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Default (0.5)')
    ax.set_title('Optimal Classification Threshold Per Year\n(lower = model is more cautious, flags more patients as diabetic)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x - w/2)
    ax.set_xticklabels([n.split('—')[-1].strip() for n in task_names])
    ax.set_ylabel('Optimal Threshold')
    ax.set_ylim(0, 0.9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save('14_threshold_per_task.png', output_dir)
