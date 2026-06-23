"""
main.py
-------
Temporal Incremental Continual Learning for Diabetes Prediction
Transformer + EWC + Replay Buffer — Three-way comparison

PIPELINE:
    Phase A — Sequential WITHOUT EWC (baseline, also provides FWT reference)
    Phase B — Sequential WITH EWC only
    Phase C — Sequential WITH EWC + Replay Buffer

    Standalone baselines removed. Phase A serves as the FWT reference:
        FWT = accuracy on Task N before it was trained (from Phase A log)
              minus the standalone accuracy approximated as Phase A Task N result.

    This is valid because Phase A trains tasks sequentially and evaluates
    all tasks at each stage — R(i-1, i) is therefore available without
    needing a separate standalone run.

DATA:
    data/brfss_2015.csv
    data/brfss_2019.csv
    data/brfss_2023.csv

MODEL:
    TabularTransformer — self-attention across features.
    Learns which feature combinations predict diabetes,
    not just individual feature weights.

HOW TO RUN:
    pip install -r requirements.txt
    python main.py
"""

from src.dataset  import load_temporal_tasks, to_dataloader
from src.model    import init_model
from src.ewc      import EWC
from src.replay   import ReplayBuffer
from src.train    import train_normal, train_ewc, train_ewc_replay
from src.evaluate import (
    evaluate, evaluate_seen_tasks, full_metrics,
    compute_backward_transfer, compute_forward_transfer,
    calibrate_all_tasks, full_metrics_calibrated,
    save_results, save_full_metrics,
    plot_final_accuracy,
    plot_ewc_accuracy_over_stages,
    plot_replay_accuracy_over_stages,
    plot_noewc_accuracy_over_stages,
    plot_backward_transfer,
    plot_forward_transfer,
    plot_transfer_summary,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_forgetting_heatmap,
    plot_training_loss_curves,
    plot_ewc_penalty_ratio,
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
EPOCHS             = 50
LR                 = 0.001
LAMBDA_EWC         = 10   # Higher lambda needed with fixed LR in EWC training
BATCH_SIZE         = 64
MAX_FISHER_SAMPLES = 5000

# Replay buffer settings
REPLAY_SAMPLES_PER_TASK = 20000   # samples stored per past task (up from 500)
REPLAY_RATIO            = 0.25   # 25% of each batch is replay data

# Transformer settings
EMBED_DIM  = 64
N_HEADS    = 4
N_LAYERS   = 2
DROPOUT    = 0.1


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
print(f"\n  Model          : TabularTransformer")
print(f"  Embed dim      : {EMBED_DIM} | Heads: {N_HEADS} | Layers: {N_LAYERS}")
print(f"  Lambda EWC     : {LAMBDA_EWC} | Replay samples: {REPLAY_SAMPLES_PER_TASK}")


# ═══════════════════════════════════════════════════════════
# PHASE A — SEQUENTIAL WITHOUT EWC
#
# One Transformer model trains through all 3 years with no
# protection. This is the forgetting baseline. Its accuracy
# log also serves as the FWT reference — R(i-1, i) comes
# from this log, so no separate standalone phase is needed.
# ═══════════════════════════════════════════════════════════
section("PHASE A: SEQUENTIAL — WITHOUT EWC (forgetting baseline)")

noewc_model     = init_model(input_size)
noewc_log       = []
noewc_histories = []

for i, name in enumerate(TASK_NAMES):
    print(f"\n  [No-EWC] Training on {name}...")
    noewc_model, h = train_normal(noewc_model, train_loaders[i], EPOCHS, LR)
    noewc_histories.append(h)
    print(f"\n  Evaluating all years after {name}:")
    noewc_log.append(evaluate_seen_tasks(noewc_model, test_loaders, TASK_NAMES, i))

# Use Phase A log as FWT baseline — accuracy on Task i before Task i was trained
# is already recorded in noewc_log[i-1][task_i_name]
baseline_accs = {name: noewc_log[i][name] for i, name in enumerate(TASK_NAMES)}
print(f"\n  FWT baselines (from Phase A): {baseline_accs}")

section("PHASE A: TRANSFER METRICS — No-EWC")
bwt_noewc, per_bwt_noewc = compute_backward_transfer(noewc_log, TASK_NAMES)
fwt_noewc, per_fwt_noewc = compute_forward_transfer(noewc_log, TASK_NAMES, baseline_accs)

print("\n  Full metrics — No-EWC final model:")
noewc_metrics = [full_metrics(noewc_model, loader, name)
                 for loader, name in zip(test_loaders, TASK_NAMES)]


# ═══════════════════════════════════════════════════════════
# PHASE B — SEQUENTIAL WITH EWC
#
# Fixed LR (no scheduler) ensures weight changes remain
# large enough for the EWC penalty to stay active.
# Lambda=10000 with normalised Fisher provides strong
# protection without freezing the model completely.
# ═══════════════════════════════════════════════════════════
section("PHASE B: SEQUENTIAL — WITH EWC (Transformer + fixed LR)")

ewc_model     = init_model(input_size)
ewc_log       = []
ewc_histories = []
ewc_objects   = []

for i, name in enumerate(TASK_NAMES):
    if i == 0:
        print(f"\n  [EWC] Training on {name} (first year — no EWC yet)...")
        ewc_model, h = train_normal(ewc_model, train_loaders[i], EPOCHS, LR)
    else:
        print(f"\n  [EWC] Training on {name} (EWC protecting {i} previous year(s))...")
        ewc_model, h = train_ewc(ewc_model, train_loaders[i],
                                  ewc_objects, LAMBDA_EWC, EPOCHS, LR)

    ewc_histories.append(h)
    print(f"\n  Evaluating all years after {name}:")
    ewc_log.append(evaluate_seen_tasks(ewc_model, test_loaders, TASK_NAMES, i))

    print(f"\n  Computing normalised Fisher for {name}...")
    ewc_objects.append(EWC(ewc_model, train_loaders[i], MAX_FISHER_SAMPLES, normalise=True))

section("PHASE B: TRANSFER METRICS — EWC")
bwt_ewc, per_bwt_ewc = compute_backward_transfer(ewc_log, TASK_NAMES)
fwt_ewc, per_fwt_ewc = compute_forward_transfer(ewc_log, TASK_NAMES, baseline_accs)

print("\n  Full metrics — EWC final model:")
ewc_metrics = [full_metrics(ewc_model, loader, name)
               for loader, name in zip(test_loaders, TASK_NAMES)]


# ═══════════════════════════════════════════════════════════
# PHASE C — SEQUENTIAL WITH EWC + REPLAY BUFFER
#
# Replay stores 2000 samples per past task and mixes 25%
# of each batch with replay data. This gives the model a
# direct reminder of past patterns alongside EWC weight
# protection — the two methods are complementary.
#
# EWC objects are recomputed fresh for this phase — the
# Transformer weights after Phase B carry different Fisher
# values than the fresh model here.
# ═══════════════════════════════════════════════════════════
section("PHASE C: SEQUENTIAL — WITH EWC + REPLAY BUFFER")

replay_model     = init_model(input_size)
replay_log       = []
replay_histories = []
replay_ewc_objects = []
replay_buffer    = ReplayBuffer(
    samples_per_task=REPLAY_SAMPLES_PER_TASK,
    replay_ratio=REPLAY_RATIO,
)

for i, name in enumerate(TASK_NAMES):
    if i == 0:
        print(f"\n  [EWC+Replay] Training on {name} (first year — no EWC yet)...")
        replay_model, h = train_normal(replay_model, train_loaders[i], EPOCHS, LR)
    else:
        print(f"\n  [EWC+Replay] Training on {name} "
              f"(EWC + {replay_buffer.total_stored:,} replay samples)...")
        replay_model, h = train_ewc_replay(
            replay_model, train_loaders[i],
            replay_ewc_objects, replay_buffer,
            LAMBDA_EWC, EPOCHS, LR,
        )

    replay_histories.append(h)
    print(f"\n  Evaluating all years after {name}:")
    replay_log.append(evaluate_seen_tasks(replay_model, test_loaders, TASK_NAMES, i))

    # Store samples from this task before moving on
    print(f"\n  Storing replay samples from {name}...")
    replay_buffer.add_task(tasks[i]['X_train'], tasks[i]['y_train'])

    print(f"\n  Computing normalised Fisher for {name}...")
    replay_ewc_objects.append(
        EWC(replay_model, train_loaders[i], MAX_FISHER_SAMPLES, normalise=True)
    )

section("PHASE C: TRANSFER METRICS — EWC + REPLAY")
bwt_replay, per_bwt_replay = compute_backward_transfer(replay_log, TASK_NAMES)
fwt_replay, per_fwt_replay = compute_forward_transfer(replay_log, TASK_NAMES, baseline_accs)

print("\n  Full metrics — EWC+Replay final model:")
replay_metrics = [full_metrics(replay_model, loader, name)
                  for loader, name in zip(test_loaders, TASK_NAMES)]


# ═══════════════════════════════════════════════════════════
# PHASE D — THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════
section("PHASE D: PER-YEAR THRESHOLD CALIBRATION")

print("\n  Calibrating — No-EWC model...")
noewc_thresholds = calibrate_all_tasks(noewc_model, test_loaders, TASK_NAMES)

print("\n  Calibrating — EWC model...")
ewc_thresholds = calibrate_all_tasks(ewc_model, test_loaders, TASK_NAMES)

print("\n  Calibrating — EWC+Replay model...")
replay_thresholds = calibrate_all_tasks(replay_model, test_loaders, TASK_NAMES)

print("\n  Calibrated metrics — No-EWC:")
noewc_metrics_cal = [
    full_metrics_calibrated(noewc_model, loader, name, noewc_thresholds[name])
    for loader, name in zip(test_loaders, TASK_NAMES)
]

print("\n  Calibrated metrics — EWC:")
ewc_metrics_cal = [
    full_metrics_calibrated(ewc_model, loader, name, ewc_thresholds[name])
    for loader, name in zip(test_loaders, TASK_NAMES)
]

print("\n  Calibrated metrics — EWC+Replay:")
replay_metrics_cal = [
    full_metrics_calibrated(replay_model, loader, name, replay_thresholds[name])
    for loader, name in zip(test_loaders, TASK_NAMES)
]


# ═══════════════════════════════════════════════════════════
# PHASE E — SAVE RESULTS AND GENERATE 14 GRAPHS
# ═══════════════════════════════════════════════════════════
section("PHASE E: SAVING RESULTS AND GENERATING 14 GRAPHS")

save_results(noewc_log,   TASK_NAMES, 'results_noewc.csv')
save_results(ewc_log,     TASK_NAMES, 'results_ewc.csv')
save_results(replay_log,  TASK_NAMES, 'results_replay.csv')
save_full_metrics(noewc_metrics,   'metrics_noewc.csv')
save_full_metrics(ewc_metrics,     'metrics_ewc.csv')
save_full_metrics(replay_metrics,  'metrics_replay.csv')

noewc_final  = noewc_log[-1]
ewc_final    = ewc_log[-1]
replay_final = replay_log[-1]

models_dict = {
    'No EWC':      noewc_model,
    'EWC':         ewc_model,
    'EWC + Replay': replay_model,
}

print("\n  Generating graphs...")
plot_final_accuracy(noewc_final, ewc_final, replay_final, TASK_NAMES)
plot_ewc_accuracy_over_stages(ewc_log, TASK_NAMES)
plot_replay_accuracy_over_stages(replay_log, TASK_NAMES)
plot_noewc_accuracy_over_stages(noewc_log, TASK_NAMES)
plot_backward_transfer(per_bwt_noewc, per_bwt_ewc, per_bwt_replay)
plot_forward_transfer(per_fwt_noewc, per_fwt_ewc, per_fwt_replay)
plot_transfer_summary(bwt_noewc, bwt_ewc, bwt_replay,
                      fwt_noewc, fwt_ewc, fwt_replay)
plot_confusion_matrices(models_dict, test_loaders, TASK_NAMES)
plot_roc_curves(models_dict, test_loaders, TASK_NAMES)
plot_metrics_comparison(noewc_metrics, ewc_metrics, replay_metrics)
plot_forgetting_heatmap(ewc_log,    TASK_NAMES, 'Forgetting Heatmap — EWC',
                        '11_heatmap_ewc.png')
plot_forgetting_heatmap(replay_log, TASK_NAMES, 'Forgetting Heatmap — EWC+Replay',
                        '12_heatmap_replay.png')
plot_training_loss_curves(ewc_histories, replay_histories, TASK_NAMES)
plot_ewc_penalty_ratio(ewc_histories, replay_histories, TASK_NAMES)


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
section("FINAL SUMMARY")

print(f"\n  Model          : TabularTransformer")
print(f"  Temporal tasks : 2015 → 2019 → 2023")
print(f"  Lambda         : {LAMBDA_EWC} | Fisher: normalised | Epochs: {EPOCHS}")
print(f"  Replay samples : {REPLAY_SAMPLES_PER_TASK} per task | Ratio: {REPLAY_RATIO}")

# Accuracy table
print(f"\n  {'Year':<30} {'No-EWC':>10} {'EWC':>10} {'EWC+Replay':>12}")
print(f"  {'-'*64}")
for name in TASK_NAMES:
    print(f"  {name:<30} "
          f"{noewc_final.get(name,0):>9.2f}% "
          f"{ewc_final.get(name,0):>9.2f}% "
          f"{replay_final.get(name,0):>11.2f}%")

print(f"\n  Backward Transfer (BWT):")
print(f"    No-EWC     : {bwt_noewc:+.2f}%")
print(f"    EWC        : {bwt_ewc:+.2f}%")
print(f"    EWC+Replay : {bwt_replay:+.2f}%  (closest to 0 = best retention)")

print(f"\n  Forward Transfer (FWT):")
print(f"    No-EWC     : {fwt_noewc:+.2f}%")
print(f"    EWC        : {fwt_ewc:+.2f}%")
print(f"    EWC+Replay : {fwt_replay:+.2f}%  (highest = best forward knowledge sharing)")

print(f"\n  Calibrated Recall — EWC:")
for m in ewc_metrics_cal:
    print(f"    {m['Task']}: Recall={m['Recall']}%  F1={m['F1']}%  (t={m['Threshold']})")

print(f"\n  Calibrated Recall — EWC+Replay:")
for m in replay_metrics_cal:
    print(f"    {m['Task']}: Recall={m['Recall']}%  F1={m['F1']}%  (t={m['Threshold']})")

print(f"\n  Graph 14 guidance (EWC penalty ratio):")
print(f"    Target: 0.5–3.0  (EWC balanced with task loss)")
print(f"    < 0.1  → increase LAMBDA_EWC")
print(f"    > 5.0  → reduce  LAMBDA_EWC")

print(f"\n  All 14 graphs and 6 CSV files saved to results/")
print("DONE.")
print("=" * 65)
