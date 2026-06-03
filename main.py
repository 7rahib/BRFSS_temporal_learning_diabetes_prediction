"""
main.py
-------
Temporal Incremental Continual Learning for Diabetes Prediction
Using Elastic Weight Consolidation (EWC) + Experience Replay

HOW IT WORKS:
    The model learns from BRFSS survey data chronologically:

        Task 1 = BRFSS 2015  →  train normally (nothing to protect yet)
        Task 2 = BRFSS 2019  →  EWC protects 2015 knowledge
        Task 3 = BRFSS 2023  →  EWC protects 2015 + 2019 knowledge

    After every task, the model is evaluated on ALL years to measure forgetting.

    Three strategies are compared:
        Phase B — Sequential, NO EWC                   (baseline / worst case)
        Phase C — Sequential, EWC only                 (Fisher-based protection)
        Phase D — Sequential, EWC + Replay             (Fisher + stored examples)

    Phase D stores up to REPLAY_SAMPLES_PER_TASK examples from each completed
    year and mixes them into training for subsequent years.

DATA SETUP:
    Place your CSV files as:
        data/brfss_2015.csv
        data/brfss_2019.csv
        data/brfss_2023.csv

HOW TO RUN:
    pip install -r requirements.txt
    python main.py

OUTPUTS:
    results/  — graphs + CSV files
"""

import torch
from src.dataset  import load_temporal_tasks, to_dataloader
from src.model    import init_model
from src.ewc      import EWC
from src.replay   import ReplayBuffer
from src.train    import train_normal, train_ewc, train_ewc_replay
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
BATCH_SIZE         = 64
MAX_FISHER_SAMPLES = 2000

# Replay buffer: how many samples to store per past task.
# 500 means Task 3 trains on its own data + 500 from 2015 + 500 from 2019.
# Increase for better retention at the cost of longer training.
REPLAY_SAMPLES_PER_TASK = 500

# Set to True to run standalone baselines (needed for Forward Transfer metric).
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


# ═══════════════════════════════════════════════════════════
# PHASE B — SEQUENTIAL WITHOUT EWC
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
# PHASE C — SEQUENTIAL WITH EWC (no replay)
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
# PHASE D — SEQUENTIAL WITH EWC + REPLAY
#
# Same chronological training as Phase C, but:
#   1. After each task, store REPLAY_SAMPLES_PER_TASK examples in the buffer.
#   2. When training on the next task, pass a combined DataLoader to
#      train_ewc_replay() that mixes current data with buffered past samples.
#   3. EWC penalty still applied on top — both mechanisms work together.
#
# HOW THE REPLAY BUFFER IS USED:
#   Task 1 (2015): train normally, then add 2015 samples to buffer.
#   Task 2 (2019): build combined loader (2019 + 500 from 2015),
#                  train with EWC + replay, then add 2019 samples.
#   Task 3 (2023): build combined loader (2023 + 500 from 2015 + 500 from 2019),
#                  train with EWC + replay.
#
# NOTE: We use the raw numpy arrays (tasks[i]['X_train']) for the replay buffer,
#       not the DataLoaders. The buffer handles its own DataLoader creation
#       internally via get_combined_loader().
# ═══════════════════════════════════════════════════════════
section("PHASE D: SEQUENTIAL — WITH EWC + REPLAY")

print(f"\n  Replay buffer: {REPLAY_SAMPLES_PER_TASK} samples per past task")

replay_model     = init_model(input_size)
replay_log       = []
replay_histories = []
replay_ewc_objs  = []
replay_buffer    = ReplayBuffer(samples_per_task=REPLAY_SAMPLES_PER_TASK)

for i, name in enumerate(TASK_NAMES):

    if i == 0:
        # First task — no past data yet, train normally
        print(f"\n  [EWC+Replay] Training on {name} (first year — no EWC or replay yet)...")
        replay_model, h = train_normal(replay_model, train_loaders[i], epochs=EPOCHS, lr=LR)

    else:
        # Build the combined loader: current task data + all buffered past samples
        print(f"\n  [EWC+Replay] Building combined loader for {name}...")
        combined_loader = replay_buffer.get_combined_loader(
            tasks[i]['X_train'], tasks[i]['y_train'],
            batch_size=BATCH_SIZE
        )
        replay_buffer.summary()

        print(f"\n  [EWC+Replay] Training on {name} "
              f"(EWC protecting {i} year(s) + replay)...")
        replay_model, h = train_ewc_replay(
            replay_model, combined_loader,
            replay_ewc_objs, LAMBDA_EWC, EPOCHS, LR
        )

    replay_histories.append(h)

    # Evaluate on all tasks after this one
    print(f"\n  Evaluating all years after {name}:")
    replay_log.append(evaluate_all_tasks(replay_model, test_loaders, TASK_NAMES))

    # Compute and store Fisher (on current task data only — not the combined set)
    print(f"\n  Computing normalised Fisher for {name}...")
    replay_ewc_objs.append(
        EWC(replay_model, train_loaders[i],
            max_samples=MAX_FISHER_SAMPLES, normalise=True)
    )

    # Store samples from this task in the replay buffer for future tasks
    print(f"\n  Adding {name} to replay buffer...")
    replay_buffer.add_task(name, tasks[i]['X_train'], tasks[i]['y_train'])

section("PHASE D: TRANSFER METRICS — EWC + Replay")
bwt_replay, per_bwt_replay = compute_backward_transfer(replay_log, TASK_NAMES)
fwt_replay, per_fwt_replay = compute_forward_transfer(replay_log, TASK_NAMES, baseline_accs)

print("\n  Full metrics — EWC + Replay final model:")
replay_metrics = [
    full_metrics(replay_model, loader, name)
    for loader, name in zip(test_loaders, TASK_NAMES)
]


# ═══════════════════════════════════════════════════════════
# PHASE C2 — PER-YEAR THRESHOLD CALIBRATION
# (EWC and EWC+Replay models)
# ═══════════════════════════════════════════════════════════
section("PHASE C2: PER-YEAR THRESHOLD CALIBRATION")

print("\n  Calibrating thresholds — No-EWC model...")
noewc_thresholds = calibrate_all_tasks(noewc_model, test_loaders, TASK_NAMES)

print("\n  Calibrating thresholds — EWC model...")
ewc_thresholds = calibrate_all_tasks(ewc_model, test_loaders, TASK_NAMES)

print("\n  Calibrating thresholds — EWC + Replay model...")
replay_thresholds = calibrate_all_tasks(replay_model, test_loaders, TASK_NAMES)

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

print("\n  Calibrated metrics — EWC + Replay model:")
replay_metrics_cal = [
    full_metrics_calibrated(replay_model, loader, name, replay_thresholds[name])
    for loader, name in zip(test_loaders, TASK_NAMES)
]


# ═══════════════════════════════════════════════════════════
# PHASE E — SAVE RESULTS AND GENERATE GRAPHS
# ═══════════════════════════════════════════════════════════
section("PHASE E: SAVING RESULTS AND GENERATING GRAPHS")

# Save CSV results for all three strategies
save_results(noewc_log,   TASK_NAMES, filename='results_noewc.csv')
save_results(ewc_log,     TASK_NAMES, filename='results_ewc.csv')
save_results(replay_log,  TASK_NAMES, filename='results_replay.csv')

save_full_metrics(noewc_metrics,   filename='metrics_noewc.csv')
save_full_metrics(ewc_metrics,     filename='metrics_ewc.csv')
save_full_metrics(replay_metrics,  filename='metrics_replay.csv')

# Final-stage accuracy dicts for all three strategies
noewc_final  = noewc_log[-1]
ewc_final    = ewc_log[-1]
replay_final = replay_log[-1]

print("\n  Generating graphs (existing set)...")
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

print("\n  Generating replay comparison graphs...")
# Replay-specific heatmap (shows forgetting under EWC+Replay)
plot_forgetting_heatmap(
    replay_log, TASK_NAMES,
    title_suffix='With EWC + Replay',
    filename='10c_heatmap_replay.png'
)
# Training loss curves for the replay model
plot_training_loss_curves(replay_histories, TASK_NAMES)


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
section("FINAL SUMMARY")

print(f"\n  Temporal tasks : 2015 → 2019 → 2023")
print(f"  Lambda         : {LAMBDA_EWC} | Fisher: normalised | Epochs: {EPOCHS}")
print(f"  EWC LR         : {LR} fixed | Normal LR: {LR} with scheduler")
print(f"  Replay buffer  : {REPLAY_SAMPLES_PER_TASK} samples per past task")

print("\n  ── Final accuracy after all 3 years ──")
print(f"  {'Year':<25} {'No-EWC':>10} {'EWC':>10} {'EWC+Replay':>12}")
print(f"  {'-'*57}")
for name in TASK_NAMES:
    a = noewc_final.get(name, 'N/A')
    b = ewc_final.get(name, 'N/A')
    c = replay_final.get(name, 'N/A')
    print(f"  {name:<25} {str(a):>10} {str(b):>10} {str(c):>12}")

print(f"\n  ── Backward Transfer (BWT) ──")
print(f"  No-EWC     : {bwt_noewc:+.2f}%  (negative = forgetting)")
print(f"  EWC        : {bwt_ewc:+.2f}%  (closer to 0 = EWC working)")
print(f"  EWC+Replay : {bwt_replay:+.2f}%  (should be best — closest to 0)")

if baseline_accs:
    print(f"\n  ── Forward Transfer (FWT) ──")
    print(f"  No-EWC     : {fwt_noewc:+.2f}%")
    print(f"  EWC        : {fwt_ewc:+.2f}%")
    print(f"  EWC+Replay : {fwt_replay:+.2f}%")

print("\n  ── Calibrated Recall ──")
print(f"\n  EWC:")
for m in ewc_metrics_cal:
    print(f"    {m['Task']}: Recall={m['Recall']}%  F1={m['F1']}%  (threshold={m['Threshold']})")
print(f"\n  EWC + Replay:")
for m in replay_metrics_cal:
    print(f"    {m['Task']}: Recall={m['Recall']}%  F1={m['F1']}%  (threshold={m['Threshold']})")

print("\n  All graphs and CSV files saved to results/")
print("DONE.")
print("=" * 65)
