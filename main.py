"""
main.py
-------
Temporal Incremental Continual Learning for Diabetes Prediction
Transformer + EWC + Replay Buffer — Three-way comparison

CHANGES IN THIS FILE (TabNet added alongside FT-Transformer):
    Phases A-E used to be flat top-level code that ran once for
    FT-Transformer. They are now inside one function, run_pipeline(),
    parameterised by which model to build and which folder to save
    results to. The MODELS list below runs that same function once for
    FT-Transformer and once for TabNet, so every graph and CSV now
    exists in two versions — one per model, saved to separate folders
    (results/fttransformer/ and results/tabnet/) so they never
    overwrite each other. No other file needed to change except the
    new src/tabnet_model.py (the TabNet model itself).

PIPELINE (run once per model):
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

MODELS:
    FT-Transformer (Feature Tokenizer + Transformer) — self-attention
    across features. Learns which feature combinations predict diabetes,
    not just individual feature weights.
    TabNet — sequential attentive feature selection (see
    src/tabnet_model.py). Run through the exact same pipeline as
    FT-Transformer for a direct, apples-to-apples comparison.

CLASS IMBALANCE:
    These CSVs are NOT pre-balanced (~15-17% diabetic prevalence in every
    year). Imbalance is handled at training time via
    BCEWithLogitsLoss(pos_weight=...), computed per-task from the training
    labels (see dataset.compute_pos_weight, POS_WEIGHT_MODE below). No
    resampling (SMOTE, WeightedRandomSampler, etc.) is done here — that's
    handled separately at the dataset level if/when needed.

    Because a 0.5 probability threshold is still a poor operating point
    under this much imbalance, Phase D calibrates a per-task threshold —
    on a held-out VALIDATION split, not the test set (see VAL_SIZE below) —
    and graphs 08b/10b (not 08/10) are the primary, reportable results.

OPTIMIZER:
    AdamW, not Adam — true decoupled weight decay, the standard choice
    for transformer-style models (see train.py).

HOW TO RUN:
    pip install -r requirements.txt
    python main.py
"""

import random
import numpy as np
import torch

from src.dataset       import load_temporal_tasks, to_dataloader, compute_pos_weight
from src.model         import init_model
from src.tabnet_model  import init_tabnet_model    # NEW — TabNet
from src.ewc           import EWC
from src.replay        import ReplayBuffer
from src.fisher_delta import (
    compute_fisher_delta, print_fisher_delta_table, save_fisher_delta,
    print_top_weight_breakdown, save_top_weight_breakdown,
)
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
    plot_confusion_matrices_calibrated,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_metrics_comparison_calibrated,
    plot_forgetting_heatmap,
    plot_training_loss_curves,
    plot_ewc_penalty_ratio,
)

# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
# Nothing was seeded before — every run had a different model weight
# initialisation, a different DataLoader shuffle order, and a different
# replay-buffer sample selection, on top of whatever config changed.
# That makes runs impossible to compare: e.g. a BWT swing between two
# runs could be a real effect of a config change, or just different
# random draws. Fixing the seed removes that source of noise so config
# changes can actually be attributed to their effect.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
EPOCHS             = 5
LR                 = 0.0015  # bumped alongside BATCH_SIZE (64->256, ~4x);
                            # Adam-family optimizers don't need full linear
                            # scaling like SGD, so ~1.5x rather than ~4x
LAMBDA_EWC         = 500    # RESET from 500 — ewc.py's Fisher normalisation
                          # changed from dividing by the global MAX to the
                          # global MEAN (see ewc.py). That makes typical
                          # Fisher values ~1000s of times larger than
                          # before, so the old LAMBDA_EWC=500 (calibrated
                          # for the old scale) would now massively
                          # overshoot. 1 is a conservative starting point,
                          # not a tuned value — check
                          # 14_ewc_penalty_ratio.png after a run (target
                          # 0.5-3.0) and adjust from here.
BATCH_SIZE         = 256  # was 64 — larger batches tend to train
                          # transformers more stably; LR bumped above to match
MAX_FISHER_SAMPLES = 5000

# Class-imbalance handling (see dataset.compute_pos_weight)
POS_WEIGHT_MODE = 'sqrt'  # 'full' = neg/pos ratio (aggressive, default).
                          # 'sqrt' = sqrt(neg/pos) (softer — trades some of
                          # the recall gain back for more precision).

# Validation split, carved out of TRAINING data (not test), used to pick
# each task's classification threshold in Phase D. Calibrating on the
# test set and then reporting metrics on that same test set would be a
# mild form of leakage — this keeps calibration and reporting separate.
VAL_SIZE = 0.15

# LR scheduler — warmup then cosine decay down to a floor (not to zero)
WARMUP_EPOCHS = None   # None = auto (10% of EPOCHS, min 1)
MIN_LR_RATIO  = 0.3    # cosine decay floor as a fraction of peak LR;
                       # kept well above 0 so EWC's penalty (which needs
                       # ongoing weight movement) stays meaningful

# Replay buffer settings
REPLAY_SAMPLES_PER_TASK = 20000   # samples stored per past task (up from 500)
REPLAY_RATIO            = 0.25   # 25% of each batch is replay data

# FT-Transformer settings
EMBED_DIM       = 64      # size of each feature's token vector
N_HEADS         = 4       # attention heads — must evenly divide EMBED_DIM
N_LAYERS        = 2       # number of stacked Transformer encoder layers
DROPOUT         = 0.1     # dropout in attention, feed-forward, and head
FFN_DIM         = None    # feed-forward hidden size; None = 4 x EMBED_DIM
ACTIVATION      = "gelu"  # feed-forward activation: "gelu" or "relu"
HEAD_HIDDEN_DIM = None    # classification head hidden size; None = EMBED_DIM // 2

# TabNet settings (NEW — see src/tabnet_model.py)
TABNET_HIDDEN_DIM = 32     # width of the feature/attentive transformers
TABNET_N_STEPS    = 3      # number of sequential decision steps
TABNET_GAMMA      = 1.5    # feature-reuse relaxation factor (higher = more reuse allowed)
TABNET_DROPOUT    = 0.1    # dropout after each feature transformer


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ─────────────────────────────────────────
# LOAD DATA (shared by both models — done once)
# ─────────────────────────────────────────
section("LOADING TEMPORAL DATA (2015 → 2019 → 2023)")

tasks, scaler, TASK_NAMES, feature_cols = load_temporal_tasks(val_size=VAL_SIZE)

train_loaders = [to_dataloader(t['X_train'], t['y_train'], BATCH_SIZE)                for t in tasks]
val_loaders   = [to_dataloader(t['X_val'],   t['y_val'],   BATCH_SIZE, shuffle=False) for t in tasks]
test_loaders  = [to_dataloader(t['X_test'],  t['y_test'],  BATCH_SIZE, shuffle=False) for t in tasks]
input_size    = tasks[0]['X_train'].shape[1]

# Per-task pos_weight for BCEWithLogitsLoss — corrects for the ~83-84%
# "always predict no diabetes" majority-class baseline (see dataset.py)
pos_weights = [compute_pos_weight(t['y_train'], mode=POS_WEIGHT_MODE) for t in tasks]

print(f"\n  Input features : {input_size}")
print(f"  Tasks          : {TASK_NAMES}")
print(f"  Pos weights    : ({POS_WEIGHT_MODE}) " +
      ", ".join(f"{n}={w.item():.2f}" for n, w in zip(TASK_NAMES, pos_weights)))
print(f"\n  Models to run  : FT-Transformer, TabNet")
print(f"  Lambda EWC     : {LAMBDA_EWC} | Replay samples: {REPLAY_SAMPLES_PER_TASK}")


# ═══════════════════════════════════════════════════════════
# ONE MODEL'S FULL PIPELINE — PHASES A THROUGH E
#
# Everything that used to be flat top-level code now lives here so it
# can be run once per model (FT-Transformer, then TabNet), each saving
# its own CSVs and graphs to its own output_dir.
# ═══════════════════════════════════════════════════════════
def run_pipeline(model_name, make_model, output_dir):
    """
    Run Phases A-E for one model and save all results under output_dir.

    model_name : label used in printouts (e.g. "FT-Transformer", "TabNet")
    make_model : zero-argument function that returns a freshly
                 initialised model for this model type
    output_dir : folder this model's CSVs and graphs are saved to, so
                 the two models' results never overwrite each other

    Returns a dict of this model's final numbers, used later for the
    side-by-side summary printed at the very end of the script.
    """
    section(f"MODEL: {model_name}")

    # ═══════════════════════════════════════════════════════════
    # PHASE A — SEQUENTIAL WITHOUT EWC
    #
    # One model trains through all 3 years with no protection. This is
    # the forgetting baseline. Its accuracy log also serves as the FWT
    # reference — R(i-1, i) comes from this log, so no separate
    # standalone phase is needed.
    # ═══════════════════════════════════════════════════════════
    section(f"[{model_name}] PHASE A: SEQUENTIAL — WITHOUT EWC (forgetting baseline)")

    noewc_model     = make_model()
    noewc_log       = []
    noewc_histories = []
    noewc_pre_accs  = {}   # zero-shot accuracy on task i, captured right before training on it

    for i, name in enumerate(TASK_NAMES):
        if i > 0:
            zero_shot_acc = evaluate(noewc_model, test_loaders[i]) * 100
            noewc_pre_accs[name] = round(zero_shot_acc, 2)
            print(f"\n  Zero-shot accuracy on {name} before training on it: {zero_shot_acc:.2f}%")

        print(f"\n  [No-EWC] Training on {name}...")
        noewc_model, h = train_normal(noewc_model, train_loaders[i], EPOCHS, LR,
                                       pos_weight=pos_weights[i],
                                       warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO)
        noewc_histories.append(h)
        print(f"\n  Evaluating all years after {name}:")
        noewc_log.append(evaluate_seen_tasks(noewc_model, test_loaders, TASK_NAMES, i))

    # FWT baseline — accuracy on Task i once it WAS fully trained (Phase A, No-EWC)
    baseline_accs = {name: noewc_log[i][name] for i, name in enumerate(TASK_NAMES)}
    print(f"\n  FWT baselines (from Phase A): {baseline_accs}")

    section(f"[{model_name}] PHASE A: TRANSFER METRICS — No-EWC")
    bwt_noewc, per_bwt_noewc = compute_backward_transfer(noewc_log, TASK_NAMES)
    fwt_noewc, per_fwt_noewc = compute_forward_transfer(noewc_pre_accs, TASK_NAMES, baseline_accs)

    print("\n  Full metrics — No-EWC final model:")
    noewc_metrics = [full_metrics(noewc_model, loader, name)
                     for loader, name in zip(test_loaders, TASK_NAMES)]

    # ═══════════════════════════════════════════════════════════
    # PHASE B — SEQUENTIAL WITH EWC
    #
    # Fixed LR (no scheduler) ensures weight changes remain large
    # enough for the EWC penalty to stay active. Lambda=10000 with
    # normalised Fisher provides strong protection without freezing
    # the model completely.
    # ═══════════════════════════════════════════════════════════
    section(f"[{model_name}] PHASE B: SEQUENTIAL — WITH EWC (fixed LR)")

    ewc_model     = make_model()
    ewc_log       = []
    ewc_histories = []
    ewc_objects   = []
    ewc_pre_accs  = {}   # zero-shot accuracy on task i, captured right before training on it

    for i, name in enumerate(TASK_NAMES):
        if i == 0:
            print(f"\n  [EWC] Training on {name} (first year — no EWC yet)...")
            ewc_model, h = train_normal(ewc_model, train_loaders[i], EPOCHS, LR,
                                         pos_weight=pos_weights[i],
                                         warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO)
        else:
            zero_shot_acc = evaluate(ewc_model, test_loaders[i]) * 100
            ewc_pre_accs[name] = round(zero_shot_acc, 2)
            print(f"\n  Zero-shot accuracy on {name} before training on it: {zero_shot_acc:.2f}%")

            print(f"\n  [EWC] Training on {name} (EWC protecting {i} previous year(s))...")
            ewc_model, h = train_ewc(ewc_model, train_loaders[i],
                                      ewc_objects, LAMBDA_EWC, EPOCHS, LR,
                                      pos_weight=pos_weights[i],
                                      warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO)

        ewc_histories.append(h)
        print(f"\n  Evaluating all years after {name}:")
        ewc_log.append(evaluate_seen_tasks(ewc_model, test_loaders, TASK_NAMES, i))

        print(f"\n  Computing normalised Fisher for {name}...")
        ewc_objects.append(EWC(ewc_model, train_loaders[i], MAX_FISHER_SAMPLES, normalise=True))

    section(f"[{model_name}] PHASE B: TRANSFER METRICS — EWC")
    bwt_ewc, per_bwt_ewc = compute_backward_transfer(ewc_log, TASK_NAMES)
    fwt_ewc, per_fwt_ewc = compute_forward_transfer(ewc_pre_accs, TASK_NAMES, baseline_accs)

    print("\n  Full metrics — EWC final model:")
    ewc_metrics = [full_metrics(ewc_model, loader, name)
                   for loader, name in zip(test_loaders, TASK_NAMES)]

    # ═══════════════════════════════════════════════════════════
    # PHASE C — SEQUENTIAL WITH EWC + REPLAY BUFFER
    #
    # Replay stores samples per past task and mixes a fraction of each
    # batch with replay data. This gives the model a direct reminder of
    # past patterns alongside EWC weight protection — the two methods
    # are complementary.
    #
    # EWC objects are recomputed fresh for this phase — the weights
    # after Phase B carry different Fisher values than the fresh model
    # here.
    # ═══════════════════════════════════════════════════════════
    section(f"[{model_name}] PHASE C: SEQUENTIAL — WITH EWC + REPLAY BUFFER")

    replay_model       = make_model()
    replay_log         = []
    replay_histories   = []
    replay_ewc_objects = []
    replay_buffer      = ReplayBuffer(
        samples_per_task=REPLAY_SAMPLES_PER_TASK,
        replay_ratio=REPLAY_RATIO,
    )

    replay_pre_accs = {}   # zero-shot accuracy on task i, captured right before training on it

    for i, name in enumerate(TASK_NAMES):
        if i == 0:
            print(f"\n  [EWC+Replay] Training on {name} (first year — no EWC yet)...")
            replay_model, h = train_normal(replay_model, train_loaders[i], EPOCHS, LR,
                                            pos_weight=pos_weights[i],
                                            warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO)
        else:
            zero_shot_acc = evaluate(replay_model, test_loaders[i]) * 100
            replay_pre_accs[name] = round(zero_shot_acc, 2)
            print(f"\n  Zero-shot accuracy on {name} before training on it: {zero_shot_acc:.2f}%")

            print(f"\n  [EWC+Replay] Training on {name} "
                  f"(EWC + {replay_buffer.total_stored:,} replay samples)...")
            replay_model, h = train_ewc_replay(
                replay_model, train_loaders[i],
                replay_ewc_objects, replay_buffer,
                LAMBDA_EWC, EPOCHS, LR,
                pos_weight=pos_weights[i],
                warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO,
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

    section(f"[{model_name}] PHASE C: TRANSFER METRICS — EWC + REPLAY")
    bwt_replay, per_bwt_replay = compute_backward_transfer(replay_log, TASK_NAMES)
    fwt_replay, per_fwt_replay = compute_forward_transfer(replay_pre_accs, TASK_NAMES, baseline_accs)

    print("\n  Full metrics — EWC+Replay final model:")
    replay_metrics = [full_metrics(replay_model, loader, name)
                      for loader, name in zip(test_loaders, TASK_NAMES)]

    # ═══════════════════════════════════════════════════════════
    # FISHER DELTA — TEMPORAL DRIFT ANALYSIS
    #
    # Compares each phase's Fisher dictionaries year-over-year to see
    # whether the model kept relying on the same weights (stable trend)
    # or shifted to different weights (drift). Does not change how
    # Fisher itself is computed (ewc.py is untouched) - this only reads
    # the .fisher dictionaries already stored inside ewc_objects /
    # replay_ewc_objects, built above in Phase B and Phase C.
    # ═══════════════════════════════════════════════════════════
    section(f"[{model_name}] FISHER DELTA: TEMPORAL DRIFT ANALYSIS")

    ewc_delta = compute_fisher_delta(ewc_objects, TASK_NAMES)
    print_fisher_delta_table(ewc_delta, "EWC only")
    save_fisher_delta(ewc_delta, 'fisher_delta_ewc.csv', output_dir=output_dir)
    print_top_weight_breakdown(ewc_delta, "EWC only")
    save_top_weight_breakdown(ewc_delta, 'fisher_delta_ewc_top_weights.csv', output_dir=output_dir)

    replay_delta = compute_fisher_delta(replay_ewc_objects, TASK_NAMES)
    print_fisher_delta_table(replay_delta, "EWC + Replay")
    save_fisher_delta(replay_delta, 'fisher_delta_replay.csv', output_dir=output_dir)
    print_top_weight_breakdown(replay_delta, "EWC + Replay")
    save_top_weight_breakdown(replay_delta, 'fisher_delta_replay_top_weights.csv', output_dir=output_dir)

    # ═══════════════════════════════════════════════════════════
    # PHASE D — THRESHOLD CALIBRATION
    # ═══════════════════════════════════════════════════════════
    section(f"[{model_name}] PHASE D: PER-YEAR THRESHOLD CALIBRATION")
    print("\n  Thresholds are picked on the VALIDATION split, then applied to the")
    print("  (unseen) test split below — avoids tuning and reporting on the same data.")

    print("\n  Calibrating — No-EWC model...")
    noewc_thresholds = calibrate_all_tasks(noewc_model, val_loaders, TASK_NAMES)

    print("\n  Calibrating — EWC model...")
    ewc_thresholds = calibrate_all_tasks(ewc_model, val_loaders, TASK_NAMES)

    print("\n  Calibrating — EWC+Replay model...")
    replay_thresholds = calibrate_all_tasks(replay_model, val_loaders, TASK_NAMES)

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
    # PHASE E — SAVE RESULTS AND GENERATE 16 GRAPHS (this model only)
    # ═══════════════════════════════════════════════════════════
    section(f"[{model_name}] PHASE E: SAVING RESULTS AND GENERATING 16 GRAPHS -> {output_dir}/")

    save_results(noewc_log,   TASK_NAMES, 'results_noewc.csv',  output_dir=output_dir)
    save_results(ewc_log,     TASK_NAMES, 'results_ewc.csv',    output_dir=output_dir)
    save_results(replay_log,  TASK_NAMES, 'results_replay.csv', output_dir=output_dir)
    save_full_metrics(noewc_metrics,   'metrics_noewc.csv',  output_dir=output_dir)
    save_full_metrics(ewc_metrics,     'metrics_ewc.csv',    output_dir=output_dir)
    save_full_metrics(replay_metrics,  'metrics_replay.csv', output_dir=output_dir)

    noewc_final  = noewc_log[-1]
    ewc_final    = ewc_log[-1]
    replay_final = replay_log[-1]

    models_dict = {
        'No EWC':      noewc_model,
        'EWC':         ewc_model,
        'EWC + Replay': replay_model,
    }

    # Per-model, per-task calibrated thresholds (from Phase D) — used to
    # produce the primary-result (08b/10b) charts below.
    thresholds_dict = {
        'No EWC':      noewc_thresholds,
        'EWC':         ewc_thresholds,
        'EWC + Replay': replay_thresholds,
    }

    print("\n  Generating graphs...")
    plot_final_accuracy(noewc_final, ewc_final, replay_final, TASK_NAMES, output_dir=output_dir)
    plot_ewc_accuracy_over_stages(ewc_log, TASK_NAMES, output_dir=output_dir)
    plot_replay_accuracy_over_stages(replay_log, TASK_NAMES, output_dir=output_dir)
    plot_noewc_accuracy_over_stages(noewc_log, TASK_NAMES, output_dir=output_dir)
    plot_backward_transfer(per_bwt_noewc, per_bwt_ewc, per_bwt_replay, output_dir=output_dir)
    plot_forward_transfer(per_fwt_noewc, per_fwt_ewc, per_fwt_replay, output_dir=output_dir)
    plot_transfer_summary(bwt_noewc, bwt_ewc, bwt_replay,
                          fwt_noewc, fwt_ewc, fwt_replay, output_dir=output_dir)
    plot_confusion_matrices(models_dict, test_loaders, TASK_NAMES, output_dir=output_dir)                                  # uncalibrated (reference)
    plot_confusion_matrices_calibrated(models_dict, test_loaders, TASK_NAMES, thresholds_dict, output_dir=output_dir)       # PRIMARY RESULT
    plot_roc_curves(models_dict, test_loaders, TASK_NAMES, output_dir=output_dir)
    plot_metrics_comparison(noewc_metrics, ewc_metrics, replay_metrics, output_dir=output_dir)                              # uncalibrated (reference)
    plot_metrics_comparison_calibrated(noewc_metrics_cal, ewc_metrics_cal, replay_metrics_cal, output_dir=output_dir)       # PRIMARY RESULT
    plot_forgetting_heatmap(ewc_log,    TASK_NAMES, 'Forgetting Heatmap — EWC',
                            '11_heatmap_ewc.png', output_dir=output_dir)
    plot_forgetting_heatmap(replay_log, TASK_NAMES, 'Forgetting Heatmap — EWC+Replay',
                            '12_heatmap_replay.png', output_dir=output_dir)
    plot_training_loss_curves(ewc_histories, replay_histories, TASK_NAMES, output_dir=output_dir)
    plot_ewc_penalty_ratio(ewc_histories, replay_histories, TASK_NAMES, output_dir=output_dir)

    # Everything the final side-by-side summary needs, for both models
    return {
        'noewc_final': noewc_final, 'ewc_final': ewc_final, 'replay_final': replay_final,
        'bwt_noewc': bwt_noewc, 'bwt_ewc': bwt_ewc, 'bwt_replay': bwt_replay,
        'fwt_noewc': fwt_noewc, 'fwt_ewc': fwt_ewc, 'fwt_replay': fwt_replay,
        'ewc_metrics_cal': ewc_metrics_cal, 'replay_metrics_cal': replay_metrics_cal,
    }


# ─────────────────────────────────────────
# RUN BOTH MODELS THROUGH THE SAME PIPELINE
# ─────────────────────────────────────────
# Each entry: (display name, zero-argument model-builder, output folder)
MODELS = [
    ("FT-Transformer",
     lambda: init_model(input_size, EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT,
                        FFN_DIM, ACTIVATION, HEAD_HIDDEN_DIM),
     "results/fttransformer"),
    ("TabNet",
     lambda: init_tabnet_model(input_size, TABNET_HIDDEN_DIM, TABNET_N_STEPS,
                               TABNET_GAMMA, TABNET_DROPOUT),
     "results/tabnet"),
]

summaries = {}
for model_name, make_model, output_dir in MODELS:
    summaries[model_name] = run_pipeline(model_name, make_model, output_dir)


# ─────────────────────────────────────────
# FINAL SUMMARY — both models, side by side
# ─────────────────────────────────────────
section("FINAL SUMMARY — FT-Transformer vs TabNet")

print(f"\n  Temporal tasks : 2015 → 2019 → 2023")
print(f"  Lambda         : {LAMBDA_EWC} | Fisher: normalised | Epochs: {EPOCHS}")
print(f"  Replay samples : {REPLAY_SAMPLES_PER_TASK} per task | Ratio: {REPLAY_RATIO}")

for model_name, s in summaries.items():
    print(f"\n  --- {model_name} ---")
    print(f"  {'Year':<30} {'No-EWC':>10} {'EWC':>10} {'EWC+Replay':>12}")
    print(f"  {'-'*64}")
    for name in TASK_NAMES:
        print(f"  {name:<30} "
              f"{s['noewc_final'].get(name,0):>9.2f}% "
              f"{s['ewc_final'].get(name,0):>9.2f}% "
              f"{s['replay_final'].get(name,0):>11.2f}%")

    print(f"\n  Backward Transfer (BWT):")
    print(f"    No-EWC     : {s['bwt_noewc']:+.2f}%")
    print(f"    EWC        : {s['bwt_ewc']:+.2f}%")
    print(f"    EWC+Replay : {s['bwt_replay']:+.2f}%  (closest to 0 = best retention)")

    print(f"\n  Forward Transfer (FWT):")
    print(f"    No-EWC     : {s['fwt_noewc']:+.2f}%")
    print(f"    EWC        : {s['fwt_ewc']:+.2f}%")
    print(f"    EWC+Replay : {s['fwt_replay']:+.2f}%  (highest = best forward knowledge sharing)")

    print(f"\n  Calibrated Recall — EWC:")
    for m in s['ewc_metrics_cal']:
        print(f"    {m['Task']}: Recall={m['Recall']}%  F1={m['F1']}%  (t={m['Threshold']})")

    print(f"\n  Calibrated Recall — EWC+Replay:")
    for m in s['replay_metrics_cal']:
        print(f"    {m['Task']}: Recall={m['Recall']}%  F1={m['F1']}%  (t={m['Threshold']})")

print(f"\n  Graph 14 guidance (EWC penalty ratio):")
print(f"    Target: 0.5–3.0  (EWC balanced with task loss)")
print(f"    < 0.1  → increase LAMBDA_EWC")
print(f"    > 5.0  → reduce  LAMBDA_EWC")

print(f"\n  Primary results are graphs 08b/10b (calibrated) — NOT 08/10 (threshold 0.5).")
print(f"  Each model's 16 graphs and 6 CSV files were saved separately:")
for model_name, _, output_dir in MODELS:
    print(f"    {model_name:<15} -> {output_dir}/")
print("DONE.")
print("=" * 65)
