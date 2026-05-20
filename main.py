"""
main.py
-------
Temporal Incremental Continual Learning for Diabetes Prediction
Using Elastic Weight Consolidation (EWC)

HOW IT WORKS:
    The model learns from BRFSS survey data chronologically:

        Task 1 = BRFSS 2015  →  train normally (nothing to protect yet)
        Task 2 = BRFSS 2019  →  EWC protects 2015 knowledge
        Task 3 = BRFSS 2023  →  EWC protects 2015 + 2019 knowledge

    After every task, the model is evaluated on ALL years to measure forgetting.

DATA SETUP:
    Place your CSV files as:
        data/brfss_2015.csv
        data/brfss_2019.csv
        data/brfss_2023.csv

HOW TO RUN:
    pip install -r requirements.txt
    python main.py

OUTPUTS:
    results/  — 14 graphs + 4 CSV files
"""

import torch
from src.dataset  import load_temporal_tasks, to_dataloader
from src.model    import init_model
from src.ewc      import EWC
from src.train    import train_normal, train_ewc
from src.evaluate import (
    evaluate, evaluate_all_tasks, full_metrics,
    compute_backward_transfer, compute_forward_transfer,
    calibrate_all_tasks, full_metrics_calibrated,
    save_results, save_full_metrics,
    plot_noewc_vs_ewc, plot_ewc_accuracy_over_stages,
    plot_noewc_accuracy_over_stages, plot_backward_transfer,
    plot_forward_transfer, plot_transfer_summary,
    plot_confusion_matrices_comparison, plot_roc_comparison,
    plot_metrics_comparison, plot_forgetting_heatmap,
    plot_training_loss_curves, plot_ewc_penalty_magnitude,
    plot_calibrated_metrics, plot_threshold_comparison,
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
EPOCHS             = 50      # training epochs per task
LR                 = 0.001   # learning rate
LAMBDA_EWC         = 10000   # EWC penalty strength
                             # Raised from 2000. The scheduler was killing
                             # weight movement, making EWC invisible even at 2000.
                             # With fixed LR in train_ewc(), weights move more,
                             # so lambda needs to be higher to constrain them.
                             # Target: EWC/Task ratio of 0.5–3.0 in graph 12.
BATCH_SIZE         = 64
MAX_FISHER_SAMPLES = 2000

# Set to True to run standalone baselines (needed for Forward Transfer metric).
# Set to False to skip baselines and go straight to sequential training —
# saves significant time if you only care about BWT and EWC vs No-EWC.
RUN_BASELINES = True


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
section("LOADING TEMPORAL DATA (2015 → 2019 → 2023)")

tasks, scaler, TASK_NAMES, feature_cols = load_temporal_tasks(apply_smote=True)

train_loaders = [to_dataloader(t['X_train'], t['y_train'], BATCH_SIZE)                for t in tasks]
test_loaders  = [to_dataloader(t['X_test'],  t['y_test'],  BATCH_SIZE, shuffle=False) for t in tasks]
input_size    = tasks[0]['X_train'].shape[1]

print(f"\n  Input features : {input_size}")
print(f"  Tasks          : {TASK_NAMES}")


# ═══════════════════════════════════════════════════════════
# PHASE A — STANDALONE BASELINES (optional)
#
# Train one fresh model per year independently.
# Gives the ceiling accuracy for each year and is used as the
# reference point for computing Forward Transfer (FWT).
#
# WHY IT EXISTS:
#   FWT = accuracy on Task N before it was trained − standalone accuracy
#   Without the standalone score, FWT cannot be computed.
#
# WHY YOU CAN SKIP IT:
#   If you only care about BWT and EWC vs No-EWC comparison, set
#   RUN_BASELINES = False. This saves ~3× the training time.
# ═══════════════════════════════════════════════════════════
baseline_accs = {}

if RUN_BASELINES:
    section("PHASE A: STANDALONE BASELINES (one fresh model per year)")
    for i, name in enumerate(TASK_NAMES):
        print(f"\n  Training standalone model for: {name}")
        m    = init_model(input_size)
        m, _ = train_normal(m, train_loaders[i], epochs=EPOCHS, lr=LR)
        acc  = evaluate(m, test_loaders[i])
        baseline_accs[name] = round(acc * 100, 2)
        print(f"  Standalone accuracy: {acc*100:.2f}%")
    print(f"\n  Baselines: {baseline_accs}")
else:
    section("PHASE A: SKIPPED (RUN_BASELINES = False)")
    print("  Forward Transfer will not be computed.")
    print("  Set RUN_BASELINES = True to enable it.")


# ═══════════════════════════════════════════════════════════
# PHASE B — SEQUENTIAL WITHOUT EWC
#
# One model trains through all three years with no protection.
# Shows catastrophic forgetting — this is the baseline that
# EWC is compared against.
# ═══════════════════════════════════════════════════════════
section("PHASE B: SEQUENTIAL — WITHOUT EWC")

noewc_model     = init_model(input_size)
noewc_log       = []
noewc_histories = []

for i, name in enumerate(TASK_NAMES):
    print(f"\n  [No-EWC] Training on {name}...")
    noewc_model, h = train_normal(noewc_model, train_loaders[i], epochs=EPOCHS, lr=LR)
    noewc_histories.append(h)
    print(f"\n  Evaluating all years after {name}:")
    noewc_log.append(evaluate_all_tasks(noewc_model, test_loaders, TASK_NAMES))

section("PHASE B: TRANSFER METRICS — No-EWC")
bwt_noewc, per_bwt_noewc = compute_backward_transfer(noewc_log, TASK_NAMES)
fwt_noewc, per_fwt_noewc = compute_forward_transfer(noewc_log, TASK_NAMES, baseline_accs)

print("\n  Full metrics — No-EWC final model:")
noewc_metrics = [
    full_metrics(noewc_model, loader, name)
    for loader, name in zip(test_loaders, TASK_NAMES)
]


# ═══════════════════════════════════════════════════════════
# PHASE C — SEQUENTIAL WITH EWC
#
# Same chronological training, but after each year:
#   1. Compute the Fisher Information Matrix
#   2. Save the current weights as θ*
#   3. Next year's training includes an EWC penalty that
#      resists changes to important weights
#
# CRITICAL FIX APPLIED:
#   train_ewc() now uses a FIXED learning rate with no scheduler.
#   Previously, ReduceLROnPlateau dropped LR to ~0.000031, which
#   made weight changes so small that (theta - theta*)^2 was
#   negligible — effectively disabling EWC regardless of lambda.
#   With fixed LR, weights move consistently and EWC stays active.
#
# LAMBDA RAISED TO 10000:
#   With fixed LR, weights now move more per epoch, so a higher
#   lambda is needed to provide meaningful resistance.
#   Target: EWC/Task ratio of 0.5–3.0 (check graph 12).
# ═══════════════════════════════════════════════════════════
section("PHASE C: SEQUENTIAL — WITH EWC (fixed LR, lambda=10000)")

ewc_model     = init_model(input_size)
ewc_log       = []
ewc_histories = []
ewc_objects   = []

for i, name in enumerate(TASK_NAMES):
    if i == 0:
        print(f"\n  [EWC] Training on {name} (first year — no EWC yet)...")
        ewc_model, h = train_normal(ewc_model, train_loaders[i], epochs=EPOCHS, lr=LR)
    else:
        print(f"\n  [EWC] Training on {name} (EWC protecting {i} previous year(s))...")
        ewc_model, h = train_ewc(
            ewc_model, train_loaders[i],
            ewc_objects, LAMBDA_EWC, EPOCHS, LR
        )

    ewc_histories.append(h)
    print(f"\n  Evaluating all years after {name}:")
    ewc_log.append(evaluate_all_tasks(ewc_model, test_loaders, TASK_NAMES))

    print(f"\n  Computing normalised Fisher for {name}...")
    ewc_objects.append(
        EWC(ewc_model, train_loaders[i],
            max_samples=MAX_FISHER_SAMPLES, normalise=True)
    )

section("PHASE C: TRANSFER METRICS — EWC")
bwt_ewc, per_bwt_ewc = compute_backward_transfer(ewc_log, TASK_NAMES)
fwt_ewc, per_fwt_ewc = compute_forward_transfer(ewc_log, TASK_NAMES, baseline_accs)

print("\n  Full metrics — EWC final model:")
ewc_metrics = [
    full_metrics(ewc_model, loader, name)
    for loader, name in zip(test_loaders, TASK_NAMES)
]


# ═══════════════════════════════════════════════════════════
# PHASE C2 — PER-YEAR THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════
section("PHASE C2: PER-YEAR THRESHOLD CALIBRATION")

print("\n  Calibrating thresholds — No-EWC model...")
noewc_thresholds = calibrate_all_tasks(noewc_model, test_loaders, TASK_NAMES)

print("\n  Calibrating thresholds — EWC model...")
ewc_thresholds = calibrate_all_tasks(ewc_model, test_loaders, TASK_NAMES)

print("\n  Calibrated metrics — No-EWC model:")
noewc_metrics_cal = [
    full_metrics_calibrated(noewc_model, loader, name, noewc_thresholds[name])
    for loader, name in zip(test_loaders, TASK_NAMES)
]

print("\n  Calibrated metrics — EWC model:")
ewc_metrics_cal = [
    full_metrics_calibrated(ewc_model, loader, name, ewc_thresholds[name])
    for loader, name in zip(test_loaders, TASK_NAMES)
]


# ═══════════════════════════════════════════════════════════
# PHASE D — GENERATE ALL GRAPHS AND SAVE RESULTS
# ═══════════════════════════════════════════════════════════
section("PHASE D: SAVING RESULTS AND GENERATING 14 GRAPHS")

save_results(noewc_log,   TASK_NAMES, filename='results_noewc.csv')
save_results(ewc_log,     TASK_NAMES, filename='results_ewc.csv')
save_full_metrics(noewc_metrics,      filename='metrics_noewc.csv')
save_full_metrics(ewc_metrics,        filename='metrics_ewc.csv')

noewc_final = noewc_log[-1]
ewc_final   = ewc_log[-1]

print("\n  Generating graphs...")
plot_noewc_vs_ewc(noewc_final, ewc_final, TASK_NAMES)
plot_ewc_accuracy_over_stages(ewc_log, TASK_NAMES)
plot_noewc_accuracy_over_stages(noewc_log, TASK_NAMES)
plot_backward_transfer(per_bwt_noewc, per_bwt_ewc)
plot_forward_transfer(per_fwt_noewc, per_fwt_ewc)
plot_transfer_summary(bwt_noewc, bwt_ewc, fwt_noewc, fwt_ewc)
plot_confusion_matrices_comparison(noewc_model, ewc_model, test_loaders, TASK_NAMES)
plot_roc_comparison(noewc_model, ewc_model, test_loaders, TASK_NAMES)
plot_metrics_comparison(noewc_metrics, ewc_metrics)
plot_forgetting_heatmap(ewc_log,   TASK_NAMES, title_suffix='With EWC',    filename='10a_heatmap_ewc.png')
plot_forgetting_heatmap(noewc_log, TASK_NAMES, title_suffix='Without EWC', filename='10b_heatmap_noewc.png')
plot_training_loss_curves(ewc_histories, TASK_NAMES)
plot_ewc_penalty_magnitude(ewc_histories, TASK_NAMES)
plot_calibrated_metrics(noewc_metrics_cal, ewc_metrics_cal)
plot_threshold_comparison(noewc_thresholds, ewc_thresholds, TASK_NAMES)


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
section("FINAL SUMMARY")

print(f"\n  Temporal tasks : 2015 → 2019 → 2023")
print(f"  Lambda         : {LAMBDA_EWC} | Fisher: normalised | Epochs: {EPOCHS}")
print(f"  EWC LR         : {LR} fixed (no scheduler) | Normal LR: {LR} with scheduler")

print("\n  NO-EWC — Final accuracy after all 3 years:")
for name, acc in noewc_final.items():
    print(f"    {name}: {acc}%")

print("\n  EWC — Final accuracy after all 3 years:")
for name, acc in ewc_final.items():
    print(f"    {name}: {acc}%")

print(f"\n  Backward Transfer (BWT):")
print(f"    No-EWC : {bwt_noewc:+.2f}%  (negative = forgetting | 0 = perfect)")
print(f"    EWC    : {bwt_ewc:+.2f}%  (closer to 0 = EWC working)")

if baseline_accs:
    print(f"\n  Forward Transfer (FWT):")
    print(f"    No-EWC : {fwt_noewc:+.2f}%")
    print(f"    EWC    : {fwt_ewc:+.2f}%")

print("\n  Calibrated Recall (EWC):")
for m in ewc_metrics_cal:
    print(f"    {m['Task']}: Recall={m['Recall']}%  F1={m['F1']}%  (threshold={m['Threshold']})")

print("\n  Graph 12 — EWC penalty ratio:")
print("    Target: 0.5–3.0  (EWC balanced with task loss)")
print("    If < 0.1  → increase LAMBDA_EWC (try 20000 or 50000)")
print("    If > 5.0  → reduce  LAMBDA_EWC (try 5000)")
print("\n  All 14 graphs and 4 CSV files saved to results/")
print("DONE.")
print("=" * 65)
