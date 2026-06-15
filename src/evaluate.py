"""
evaluate.py
-----------
All evaluation metrics and graphs for temporal continual learning.

THREE-WAY COMPARISON:
    All graphs compare No-EWC vs EWC vs EWC+Replay side by side.

METRICS:
    Accuracy   — misleading with class imbalance; included for completeness
    Recall     — % of actual diabetics caught; the primary clinical metric
    Precision  — % of positive predictions that are correct
    F1 Score   — harmonic mean of precision and recall; best single metric
    AUC-ROC    — threshold-independent discriminative ability
    BWT        — backward transfer: how much later tasks hurt earlier ones
    FWT        — forward transfer: did earlier tasks help later ones
                 (computed using Phase A No-EWC as the baseline)

GRAPHS (14 total):
    01 — Final accuracy: No-EWC vs EWC vs EWC+Replay
    02 — EWC accuracy over training stages
    03 — EWC+Replay accuracy over training stages
    04 — No-EWC forgetting over training stages
    05 — Backward transfer per year (all three methods)
    06 — Forward transfer per year (all three methods)
    07 — Transfer summary (overall BWT + FWT)
    08 — Confusion matrices: No-EWC vs EWC vs EWC+Replay
    09 — ROC curves: all three methods
    10 — Precision / Recall / F1: all three methods
    11 — Forgetting heatmap: EWC
    12 — Forgetting heatmap: EWC+Replay
    13 — Training loss curves (task loss vs EWC penalty)
    14 — EWC penalty ratio (diagnoses lambda issues)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

os.makedirs('results', exist_ok=True)

COLORS = {
    'noewc':  '#EF5350',
    'ewc':    '#42A5F5',
    'replay': '#66BB6A',
}


# ──────────────────────────────────────────
# CORE PREDICTION UTILITIES
# ──────────────────────────────────────────

def get_predictions(model, dataloader):
    """Run model. Returns (labels, predictions, probabilities)."""
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            out = model(X).reshape(-1)
            labels_all.extend(y.cpu().numpy())
            preds_all.extend((out >= 0.5).float().cpu().numpy())
            probs_all.extend(out.cpu().numpy())
    return np.array(labels_all), np.array(preds_all), np.array(probs_all)


def evaluate(model, dataloader):
    """Return accuracy (0.0 to 1.0)."""
    labels, preds, _ = get_predictions(model, dataloader)
    return accuracy_score(labels, preds)


def full_metrics(model, dataloader, task_name):
    """Compute and print all metrics. Returns dict."""
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
    print(f"    Recall    : {rec*100:.2f}%  <- most important clinically")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    AUC-ROC   : {auc:.4f}")
    return {'Task': task_name, 'Accuracy': round(acc*100,2),
            'Precision': round(prec*100,2), 'Recall': round(rec*100,2),
            'F1': round(f1*100,2), 'AUC': round(float(auc),4)}


def evaluate_all_tasks(model, test_loaders, task_names):
    """Evaluate on all tasks. Returns {name: accuracy%}."""
    results = {}
    for name, loader in zip(task_names, test_loaders):
        acc = evaluate(model, loader)
        results[name] = round(acc * 100, 2)
        print(f"    {name}: {acc*100:.2f}%")
    return results


# ──────────────────────────────────────────
# CONTINUAL LEARNING METRICS
# ──────────────────────────────────────────

def compute_backward_transfer(results_log, task_names):
    """
    BWT = average [ R(final, i) - R(i, i) ] for all tasks except last.

    R(i, i)     = accuracy on task i right after it was trained
    R(final, i) = accuracy on task i after all tasks trained

    BWT = 0   → perfect retention
    BWT < 0   → forgetting
    BWT > 0   → accuracy improved (check recall — may be threshold shift)
    """
    final        = results_log[-1]
    per_task_bwt = {}
    for i, name in enumerate(task_names[:-1]):
        a = results_log[i].get(name)
        b = final.get(name)
        if a is not None and b is not None:
            per_task_bwt[name] = round(b - a, 2)

    overall = round(np.mean(list(per_task_bwt.values())), 2) if per_task_bwt else 0.0
    print(f"\n  Backward Transfer (BWT): {overall:+.2f}%")
    for name, val in per_task_bwt.items():
        print(f"    {name}: {val:+.2f}%  {'✓' if val >= -2 else '✗'}")
    return overall, per_task_bwt


def compute_forward_transfer(results_log, task_names, baseline_accs):
    """
    FWT = average [ R(i-1, i) - baseline_i ] for tasks i > 1.

    R(i-1, i)  = accuracy on task i measured before task i was trained
    baseline_i = No-EWC standalone accuracy for task i (Phase A)

    Positive FWT → earlier years helped future years.
    """
    per_task_fwt = {}
    for i in range(1, len(task_names)):
        name = task_names[i]
        pre  = results_log[i - 1].get(name)
        base = baseline_accs.get(name)
        if pre is not None and base is not None:
            per_task_fwt[name] = round(pre - base, 2)

    overall = round(np.mean(list(per_task_fwt.values())), 2) if per_task_fwt else 0.0
    print(f"\n  Forward Transfer (FWT): {overall:+.2f}%")
    for name, val in per_task_fwt.items():
        print(f"    {name}: {val:+.2f}%  {'✓ positive' if val >= 0 else '✗ negative'}")
    return overall, per_task_fwt


# ──────────────────────────────────────────
# THRESHOLD CALIBRATION
# ──────────────────────────────────────────

def find_best_threshold(model, dataloader):
    """Find threshold maximising F1. Returns (threshold, best_f1%)."""
    labels, _, probs = get_predictions(model, dataloader)
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.90, 0.05):
        score = f1_score(labels, (probs >= t).astype(float), zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t
    return round(float(best_thresh), 2), round(best_f1 * 100, 2)


def calibrate_all_tasks(model, test_loaders, task_names):
    """Find best threshold per task. Returns {name: threshold}."""
    print("\n  Per-task threshold calibration:")
    thresholds = {}
    for name, loader in zip(task_names, test_loaders):
        t, f1 = find_best_threshold(model, loader)
        thresholds[name] = t
        print(f"    {name}: threshold={t:.2f}  (best F1={f1:.1f}%)")
    return thresholds


def full_metrics_calibrated(model, dataloader, task_name, threshold):
    """Compute metrics using a task-specific threshold."""
    labels, _, probs = get_predictions(model, dataloader)
    preds = (probs >= threshold).astype(float)
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    f1    = f1_score(labels, preds, zero_division=0)
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
    return {'Task': task_name, 'Threshold': threshold,
            'Accuracy': round(acc*100,2), 'Precision': round(prec*100,2),
            'Recall': round(rec*100,2), 'F1': round(f1*100,2),
            'AUC': round(float(auc),4)}


# ──────────────────────────────────────────
# SAVE UTILITIES
# ──────────────────────────────────────────

def save_results(results_log, task_names, filename, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    stages = [f"After Task {i+1}" for i in range(len(results_log))]
    rows   = [{'Stage': s, **r} for s, r in zip(stages, results_log)]
    pd.DataFrame(rows).to_csv(f"{output_dir}/{filename}", index=False)
    print(f"  Saved: {output_dir}/{filename}")


def save_full_metrics(metrics_list, filename, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(metrics_list).to_csv(f"{output_dir}/{filename}", index=False)
    print(f"  Saved: {output_dir}/{filename}")


def _save(filename, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def _short(name):
    """Shorten task name for axis labels."""
    return name.split('—')[-1].strip()


# ──────────────────────────────────────────
# GRAPH 01 — Final Accuracy Comparison
# ──────────────────────────────────────────

def plot_final_accuracy(noewc_final, ewc_final, replay_final, task_names, output_dir='results'):
    """Three-way bar chart: final accuracy after all tasks trained."""
    labels = [_short(n) for n in task_names]
    x      = np.arange(len(labels))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, vals, label, col in [
        (-w, [noewc_final.get(n,0)  for n in task_names], 'No EWC',      COLORS['noewc']),
        ( 0, [ewc_final.get(n,0)    for n in task_names], 'EWC',         COLORS['ewc']),
        ( w, [replay_final.get(n,0) for n in task_names], 'EWC + Replay',COLORS['replay']),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label, color=col, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Final Accuracy After All 3 Temporal Tasks\n(No-EWC vs EWC vs EWC+Replay)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 115)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save('01_final_accuracy.png', output_dir)


# ──────────────────────────────────────────
# GRAPHS 02–04 — Accuracy Over Stages
# ──────────────────────────────────────────

def _plot_stages(results_log, task_names, title, filename, colors, linestyle='-', output_dir='results'):
    stages = [f"After\n{_short(n)}" for n in task_names]
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(task_names):
        vals = [r[name] for r in results_log if name in r]
        xpts = stages[:len(vals)]
        plt.plot(xpts, vals, marker='o', linewidth=2.5, markersize=9,
                 label=name, color=colors[i % len(colors)], linestyle=linestyle)
        for x, y in zip(xpts, vals):
            plt.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                         xytext=(0, 10), ha='center', fontsize=9)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Training Stage')
    plt.ylabel('Accuracy (%)')
    plt.ylim(50, 110)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save(filename, output_dir)


def plot_ewc_accuracy_over_stages(results_log, task_names, output_dir='results'):
    _plot_stages(results_log, task_names,
                 'EWC — Accuracy on Each Year Over Stages',
                 '02_ewc_accuracy_stages.png',
                 ['#2196F3', '#4CAF50', '#FF5722'], output_dir=output_dir)


def plot_replay_accuracy_over_stages(results_log, task_names, output_dir='results'):
    _plot_stages(results_log, task_names,
                 'EWC + Replay — Accuracy on Each Year Over Stages',
                 '03_replay_accuracy_stages.png',
                 ['#1565C0', '#2E7D32', '#BF360C'], output_dir=output_dir)


def plot_noewc_accuracy_over_stages(results_log, task_names, output_dir='results'):
    _plot_stages(results_log, task_names,
                 'WITHOUT EWC — Temporal Forgetting In Action',
                 '04_noewc_forgetting.png',
                 ['#EF5350', '#FF7043', '#FFCA28'],
                 linestyle='--', output_dir=output_dir)


# ──────────────────────────────────────────
# GRAPHS 05–07 — Transfer Metrics
# ──────────────────────────────────────────

def plot_backward_transfer(per_bwt_noewc, per_bwt_ewc, per_bwt_replay, output_dir='results'):
    """Three-way BWT bar chart."""
    tasks  = list(per_bwt_ewc.keys())
    x      = np.arange(len(tasks))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, vals, label, col in [
        (-w, [per_bwt_noewc.get(t,0)  for t in tasks], 'No EWC',      COLORS['noewc']),
        ( 0, [per_bwt_ewc.get(t,0)    for t in tasks], 'EWC',         COLORS['ewc']),
        ( w, [per_bwt_replay.get(t,0) for t in tasks], 'EWC + Replay',COLORS['replay']),
    ]:
        ax.bar(x + offset, vals, w, label=label, color=col, alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Backward Transfer (BWT)\n(0% = perfect retention | negative = forgetting)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_short(t) for t in tasks])
    ax.set_ylabel('BWT (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save('05_backward_transfer.png', output_dir)


def plot_forward_transfer(per_fwt_noewc, per_fwt_ewc, per_fwt_replay, output_dir='results'):
    """Three-way FWT bar chart."""
    tasks = list(per_fwt_ewc.keys())
    if not tasks:
        return
    x = np.arange(len(tasks))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, vals, label, col in [
        (-w, [per_fwt_noewc.get(t,0)  for t in tasks], 'No EWC',      COLORS['noewc']),
        ( 0, [per_fwt_ewc.get(t,0)    for t in tasks], 'EWC',         COLORS['ewc']),
        ( w, [per_fwt_replay.get(t,0) for t in tasks], 'EWC + Replay',COLORS['replay']),
    ]:
        ax.bar(x + offset, vals, w, label=label, color=col, alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Forward Transfer (FWT)\n(positive = earlier years helped future years)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_short(t) for t in tasks])
    ax.set_ylabel('FWT (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save('06_forward_transfer.png', output_dir)


def plot_transfer_summary(bwt_noewc, bwt_ewc, bwt_replay,
                          fwt_noewc, fwt_ewc, fwt_replay, output_dir='results'):
    """Overall BWT and FWT summary for all three methods."""
    metrics = ['Backward Transfer (BWT)', 'Forward Transfer (FWT)']
    x       = np.arange(len(metrics))
    w       = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    for offset, vals, label, col in [
        (-w, [bwt_noewc,  fwt_noewc],  'No EWC',      COLORS['noewc']),
        ( 0, [bwt_ewc,    fwt_ewc],    'EWC',         COLORS['ewc']),
        ( w, [bwt_replay, fwt_replay], 'EWC + Replay',COLORS['replay']),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label, color=col, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + (0.2 if h >= 0 else -0.8),
                    f'{h:+.2f}%', ha='center', fontsize=10, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Transfer Learning Summary\n(BWT: closer to 0 | FWT: higher is better)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save('07_transfer_summary.png', output_dir)


# ──────────────────────────────────────────
# GRAPH 08 — Confusion Matrices
# ──────────────────────────────────────────

def plot_confusion_matrices(models_dict, test_loaders, task_names, output_dir='results'):
    """
    Three rows (No-EWC / EWC / EWC+Replay) × N columns (tasks).
    models_dict = {'No EWC': model, 'EWC': model, 'EWC + Replay': model}
    """
    n_models = len(models_dict)
    n_tasks  = len(task_names)
    fig, axes = plt.subplots(n_models, n_tasks, figsize=(5 * n_tasks, 5 * n_models))

    for row, (label, model) in enumerate(models_dict.items()):
        for col, (loader, name) in enumerate(zip(test_loaders, task_names)):
            ax     = axes[row][col]
            labels, preds, _ = get_predictions(model, loader)
            cm     = confusion_matrix(labels, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Diab.', 'Diab.'],
                        yticklabels=['No Diab.', 'Diab.'])
            ax.set_title(f'{label}\n{_short(name)}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrices — No EWC / EWC / EWC+Replay', fontsize=14, fontweight='bold')
    _save('08_confusion_matrices.png', output_dir)


# ──────────────────────────────────────────
# GRAPH 09 — ROC Curves
# ──────────────────────────────────────────

def plot_roc_curves(models_dict, test_loaders, task_names, output_dir='results'):
    """ROC curves for all tasks and all three methods."""
    task_colors = ['#1565C0', '#2E7D32', '#B71C1C']
    styles      = {'No EWC': '--', 'EWC': '-', 'EWC + Replay': '-.'}

    plt.figure(figsize=(9, 7))
    for loader, name, tc in zip(test_loaders, task_names, task_colors):
        for label, model in models_dict.items():
            labs, _, probs = get_predictions(model, loader)
            try:
                fpr, tpr, _ = roc_curve(labs, probs)
                auc         = roc_auc_score(labs, probs)
                plt.plot(fpr, tpr, linestyle=styles[label], linewidth=1.8,
                         color=tc, label=f'{_short(name)} {label} (AUC={auc:.3f})')
            except Exception:
                pass

    plt.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Random')
    plt.title('ROC Curves — All Methods and Years', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(fontsize=8, loc='lower right')
    plt.grid(True, alpha=0.3)
    _save('09_roc_curves.png', output_dir)


# ──────────────────────────────────────────
# GRAPH 10 — Precision / Recall / F1
# ──────────────────────────────────────────

def plot_metrics_comparison(noewc_metrics, ewc_metrics, replay_metrics, output_dir='results'):
    """Three grouped bar charts: Precision / Recall / F1 for all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, metric in zip(axes, ['Precision', 'Recall', 'F1']):
        labels  = [_short(m['Task']) for m in ewc_metrics]
        x       = np.arange(len(labels))
        w       = 0.25
        for offset, metrics_list, label, col in [
            (-w, noewc_metrics,  'No EWC',      COLORS['noewc']),
            ( 0, ewc_metrics,    'EWC',         COLORS['ewc']),
            ( w, replay_metrics, 'EWC + Replay',COLORS['replay']),
        ]:
            ax.bar(x + offset, [m[metric] for m in metrics_list], w,
                   label=label, color=col, alpha=0.85)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_ylabel('Score (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Precision / Recall / F1 — All Three Methods', fontsize=14, fontweight='bold')
    _save('10_precision_recall_f1.png', output_dir)


# ──────────────────────────────────────────
# GRAPHS 11–12 — Forgetting Heatmaps
# ──────────────────────────────────────────

def plot_forgetting_heatmap(results_log, task_names, title, filename, output_dir='results'):
    stages = [f"After Task {i+1}" for i in range(len(results_log))]
    data   = [[r.get(n, float('nan')) for n in task_names] for r in results_log]
    df     = pd.DataFrame(data, index=stages, columns=task_names)
    plt.figure(figsize=(10, 5))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlGn',
                mask=df.isna(), linewidths=0.5, linecolor='grey',
                vmin=50, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
    plt.title(title, fontsize=13, fontweight='bold')
    _save(filename, output_dir)


# ──────────────────────────────────────────
# GRAPHS 13–14 — Training Diagnostics
# ──────────────────────────────────────────

def plot_training_loss_curves(ewc_histories, replay_histories, task_names, output_dir='results'):
    """Training loss curves for EWC and EWC+Replay side by side."""
    n = len(task_names)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))

    for i, name in enumerate(task_names):
        for col, (history, label) in enumerate([
            (ewc_histories[i],    'EWC'),
            (replay_histories[i], 'EWC + Replay'),
        ]):
            ax = axes[i][col]
            epochs     = [h['epoch']      for h in history]
            total_loss = [h['total_loss'] for h in history]
            task_loss  = [h['task_loss']  for h in history]
            ewc_loss   = [h['ewc_loss']   for h in history]
            ax.plot(epochs, total_loss, 'k-',  linewidth=2,   label='Total')
            ax.plot(epochs, task_loss,  'b--', linewidth=1.8, label='Task')
            ax.plot(epochs, ewc_loss,   'r:',  linewidth=1.8, label='EWC Penalty')
            ax.set_title(f'{label} — {_short(name)}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves — EWC vs EWC+Replay', fontsize=14, fontweight='bold')
    _save('13_training_loss_curves.png', output_dir)


def plot_ewc_penalty_ratio(ewc_histories, replay_histories, task_names, output_dir='results'):
    """
    EWC penalty / task loss ratio.
    Target: 0.5–3.0. Below 0.1 = EWC too weak. Above 5.0 = too strong.
    """
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, histories, label in [
        (axes[0], ewc_histories,    'EWC'),
        (axes[1], replay_histories, 'EWC + Replay'),
    ]:
        for history, name, color in zip(histories, task_names, colors):
            if all(h['ewc_loss'] == 0 for h in history):
                continue
            epochs = [h['epoch'] for h in history]
            ratio  = [h['ewc_loss'] / (h['task_loss'] + 1e-8) for h in history]
            ax.plot(epochs, ratio, linewidth=2, label=_short(name), color=color)

        ax.axhline(0.5, color='green', linestyle=':', linewidth=1, label='Min target (0.5)')
        ax.axhline(3.0, color='red',   linestyle=':', linewidth=1, label='Max target (3.0)')
        ax.set_title(f'{label} — Penalty Ratio\n(target: 0.5–3.0)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('EWC / Task Loss Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('EWC Penalty Ratio — Higher = EWC more active', fontsize=13, fontweight='bold')
    _save('14_ewc_penalty_ratio.png', output_dir)
