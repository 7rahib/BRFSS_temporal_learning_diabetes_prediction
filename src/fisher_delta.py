"""
fisher_delta.py
---------------
Fisher Delta — temporal drift analysis between consecutive years.

This file does NOT change how Fisher Information is computed — see
ewc.py for that (untouched). It only READS the `.fisher` dictionary
that already exists inside each already-built EWC object (one EWC
object exists per year, per phase — see main.py) and compares them
pairwise, year by year.

WHAT "DRIFT" MEANS HERE:
    Each year's Fisher dictionary says "which weights mattered for that
    year's predictions." Comparing two years' Fisher dictionaries tells
    us whether the model relied on the SAME weights in both years
    (stable / similar trend) or on DIFFERENT weights (temporal drift).

TWO METRICS PER YEAR-TRANSITION:

    1. Trend Correlation (Pearson correlation coefficient)
       Range: -1 to +1.
           +1  -> the weights that were important in year A were also
                  important in year B, in the same relative proportions
                  (SIMILAR / STABLE trend)
            0  -> no relationship between which weights mattered in
                  year A vs year B
           -1  -> weights that were important in year A tended to be
                  UNimportant in year B, and vice versa
                  (TREND CHANGED / reversed)
       This is the metric that can legitimately go negative — raw
       Fisher values are always >= 0 (they're squared gradients), so a
       plain subtraction or cosine similarity can't produce a
       meaningful negative "trend changed" signal on its own. Pearson
       correlation first subtracts each vector's own mean before
       comparing, which is what allows a genuine negative result when
       the pattern of importance flips.

    2. Magnitude Change (mean absolute difference)
       A plain, always-positive number: on average, how much did each
       weight's Fisher value change between the two years, regardless
       of whether the overall pattern (metric 1) stayed similar or not.
       A high Trend Correlation with a high Magnitude Change would mean
       "same weights mattered, but their importance grew/shrank a lot."

    Also reported: which single named parameter (layer) changed the
    most between the two years, and by how much — the "biggest factor"
    driving that year-transition's drift.
"""

import torch
import numpy as np
import pandas as pd
import os


def _flatten_fisher(fisher_dict):
    """
    Turn a {parameter_name: tensor} Fisher dictionary into one long,
    flat 1-D tensor, so every weight in the whole network can be
    compared as a single vector. `sorted()` guarantees the same
    parameter order every time, so position i in year A's flattened
    vector always lines up with position i in year B's.
    """
    names = sorted(fisher_dict.keys())
    flat  = torch.cat([fisher_dict[n].flatten() for n in names])
    return flat


def _pearson_correlation(vector_a, vector_b):
    """
    Pearson correlation coefficient between two 1-D tensors.

    Steps:
        1. Subtract each vector's own mean ("centering") — this is the
           step that allows the result to go negative.
        2. Multiply the two centered vectors element-by-element and
           sum -> tells us whether they tend to move in the same
           direction (positive contributions) or opposite directions
           (negative contributions).
        3. Divide by the product of each vector's own spread (its
           standard deviation, computed manually here as the square
           root of its summed squared deviations) -> rescales the
           result to always land between -1 and +1, regardless of the
           raw scale of the Fisher values.
    """
    a_centered = vector_a - vector_a.mean()
    b_centered = vector_b - vector_b.mean()

    numerator   = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered ** 2).sum()) * torch.sqrt((b_centered ** 2).sum())

    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()


def _biggest_layer_change(fisher_a, fisher_b):
    """
    For each named parameter (layer) separately, compute the mean
    absolute difference between its Fisher values in year A vs year B.
    Returns the name of the single layer with the largest change, and
    that change's size — i.e. "which part of the network drifted the
    most between these two years."
    """
    layer_changes = {}
    for name in fisher_a:
        diff = (fisher_a[name] - fisher_b[name]).abs().mean().item()
        layer_changes[name] = diff

    biggest_name  = max(layer_changes, key=layer_changes.get)
    biggest_value = layer_changes[biggest_name]
    return biggest_name, biggest_value


def _top_individual_weight_changes(fisher_a, fisher_b, top_n=5):
    """
    _biggest_layer_change() reports one AVERAGE number per whole named
    tensor (e.g. "head.3.weight changed by 130 on average"). That
    average can be misleading — it doesn't tell you whether every
    weight in that tensor shifted a bit, or whether just one or two
    extreme weights are dragging the average way up.

    This function instead looks INSIDE every tensor, at every single
    individual scalar weight across the whole network, and returns the
    `top_n` individual weights (with their exact position) that
    changed the most between year A and year B. This is the
    "zoomed-in" view that lets you say, concretely, "these specific
    3 weights are responsible for almost all of the drift."

    How it works, per named parameter tensor:
        1. `.abs()` — take the absolute (unsigned) difference between
           year A's and year B's Fisher values for every weight in
           this tensor, so we're only measuring size of change.
        2. `.flatten()` — turn the (possibly multi-dimensional) tensor
           of differences into a simple 1-D list, so it can be ranked.
        3. `torch.topk(flat_diff, k)` — PyTorch's built-in function
           for "give me the k largest values in this list, and their
           positions" — much faster than sorting the whole tensor
           when you only need a handful of top entries.
        4. `np.unravel_index(...)` — topk gives back a position in
           the FLATTENED 1-D list (e.g. "position 57"), but that's not
           human-readable for a multi-dimensional weight tensor.
           unravel_index converts that flat position back into its
           original row/column position (e.g. "row 3, column 9"), so
           the reported weight can actually be located again later.

    A small candidate list (top_n per tensor) is collected from every
    layer first, then re-sorted together, so the final top_n reflects
    the biggest movers across the ENTIRE network, not just within one
    layer.
    """
    candidates = []
    for name in fisher_a:
        diff      = (fisher_a[name] - fisher_b[name]).abs()
        flat_diff = diff.flatten()
        k         = min(top_n, flat_diff.numel())
        top_vals, top_idx = torch.topk(flat_diff, k)

        for value, flat_index in zip(top_vals.tolist(), top_idx.tolist()):
            position = [int(p) for p in np.unravel_index(flat_index, diff.shape)]
            label    = f"{name}{position}"
            candidates.append((label, value))

    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[:top_n]


def compute_fisher_delta(ewc_objects, task_names, top_n=5):
    """
    Computes Fisher Delta between every pair of CONSECUTIVE years, for
    ONE phase at a time.

    Call this once with `ewc_objects` (Phase B, EWC only) and again
    with `replay_ewc_objects` (Phase C, EWC+Replay) — they are kept
    separate because the two phases' models diverge, so their Fisher
    values are not directly comparable to each other, only within the
    same phase across years.

    Arguments:
        ewc_objects : list of EWC objects, in the same chronological
                      order as task_names. Each one already has
                      `.fisher` computed by ewc.py — nothing about
                      Fisher itself is recalculated here, this only
                      reads what's already stored.
        task_names  : list of year labels, e.g. ["2015", "2019", "2023"]

    Returns:
        A list of dictionaries, one per year-transition, ready to be
        printed or saved as a table.
    """
    results = []

    for i in range(len(ewc_objects) - 1):
        year_a, year_b   = task_names[i], task_names[i + 1]
        fisher_a         = ewc_objects[i].fisher
        fisher_b         = ewc_objects[i + 1].fisher

        flat_a = _flatten_fisher(fisher_a)
        flat_b = _flatten_fisher(fisher_b)

        trend_correlation = _pearson_correlation(flat_a, flat_b)
        magnitude_change  = (flat_a - flat_b).abs().mean().item()
        biggest_layer, biggest_change = _biggest_layer_change(fisher_a, fisher_b)
        top_weights = _top_individual_weight_changes(fisher_a, fisher_b, top_n=top_n)

        trend_label = "Similar (stable)" if trend_correlation > 0 else "Changed (drift)"

        results.append({
            'Transition':        f"{year_a} -> {year_b}",
            'Trend Correlation': round(trend_correlation, 4),
            'Trend':             trend_label,
            'Magnitude Change':  round(magnitude_change, 6),
            'Biggest Factor':    biggest_layer,
            'Biggest Change':    round(biggest_change, 6),
            'Top Weights':       top_weights,
        })

    return results


def print_top_weight_breakdown(results, phase_label, top_n=5):
    """
    For each year-transition, prints the individual weights (not whole
    layers) that changed the most. This is the "zoom in" companion to
    print_fisher_delta_table() — it answers "is the layer-level change
    coming from many weights moving a little, or one or two extreme
    weights moving a lot?"
    """
    print(f"\n  Top {top_n} Individual Weight Changes — {phase_label}")
    for r in results:
        print(f"\n    {r['Transition']}:")
        for rank, (label, value) in enumerate(r['Top Weights'], start=1):
            print(f"      {rank}. {label:<45} change={value:.4f}")


def save_top_weight_breakdown(results, filename, output_dir='results'):
    """
    Saves the individual-weight breakdown as its own CSV — one row per
    (transition, rank). Kept separate from save_fisher_delta() because
    the main Fisher Delta table is one row per transition, while this
    is naturally one row per individual weight.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for r in results:
        for rank, (label, value) in enumerate(r['Top Weights'], start=1):
            rows.append({
                'Transition': r['Transition'],
                'Rank':       rank,
                'Weight':     label,
                'Change':     round(value, 6),
            })
    pd.DataFrame(rows).to_csv(f"{output_dir}/{filename}", index=False)
    print(f"  Saved: {output_dir}/{filename}")


def print_fisher_delta_table(results, phase_label):
    """Prints the Fisher Delta results as a simple, readable table."""
    print(f"\n  Fisher Delta — {phase_label}")
    print(f"  {'-' * 110}")
    print(f"  {'Transition':<15}{'Correlation':>13}{'Trend':>20}"
          f"{'Magnitude':>13}{'Biggest Factor':>42}{'Change':>10}")
    print(f"  {'-' * 110}")
    for r in results:
        print(f"  {r['Transition']:<15}"
              f"{r['Trend Correlation']:>13}"
              f"{r['Trend']:>20}"
              f"{r['Magnitude Change']:>13}"
              f"{r['Biggest Factor'][:40]:>42}"
              f"{r['Biggest Change']:>10}")


def save_fisher_delta(results, filename, output_dir='results'):
    """Saves the Fisher Delta table to a CSV file, same convention as
    save_results() / save_full_metrics() in evaluate.py.

    'Top Weights' is deliberately dropped here — it's a list of
    (name, value) tuples, one row per transition wouldn't display it
    cleanly in a flat CSV. Use save_top_weight_breakdown() for that
    detail instead, saved as its own file.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = [{k: v for k, v in r.items() if k != 'Top Weights'} for r in results]
    pd.DataFrame(rows).to_csv(f"{output_dir}/{filename}", index=False)
    print(f"  Saved: {output_dir}/{filename}")
