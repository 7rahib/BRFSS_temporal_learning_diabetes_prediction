import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def permutation_importance(model, X, y, w, n_repeats=3, max_rows=20000, seed=42):
    """How much the (weighted) AUC drops when one feature's column is shuffled."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:                          # subsample to keep it fast
        keep = rng.choice(len(X), max_rows, replace=False)
        X, y, w = X[keep], y[keep], w[keep]
    model.eval()

    def weighted_auc(X_):
        with torch.no_grad():
            logits = torch.cat([model(torch.tensor(X_[i:i + 4096], dtype=torch.float32)).reshape(-1)
                                for i in range(0, len(X_), 4096)])
        return roc_auc_score(y, torch.sigmoid(logits).numpy(), sample_weight=w)

    base  = weighted_auc(X)
    drops = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[:, j] = rng.permutation(X_shuffled[:, j])
            drops[j] += (base - weighted_auc(X_shuffled)) / n_repeats
    return drops


def save_feature_drift(importance_by_year, feature_cols, label, output_dir='results', top_n=5):
    """Turns AUC drops into % shares per year, then year-to-year changes in share."""
    years  = [name.split('—')[-1].strip() for name in importance_by_year]
    drops  = pd.DataFrame(list(importance_by_year.values()), index=years, columns=feature_cols).T
    shares = drops.clip(lower=0)
    shares = shares / shares.sum() * 100           # each feature's % of total reliance
    changes = shares.diff(axis=1).iloc[:, 1:]      # this year's share minus last year's
    changes.columns = [f"{a} -> {b}" for a, b in zip(years[:-1], years[1:])]

    os.makedirs(output_dir, exist_ok=True)
    drops.round(4).to_csv(f"{output_dir}/feature_auc_drop_{label}.csv")
    shares.round(2).to_csv(f"{output_dir}/feature_share_{label}.csv")
    changes.round(2).to_csv(f"{output_dir}/feature_share_change_{label}.csv")

    print(f"\n  Top features by reliance share (%) — {label}")
    print(shares.round(1).sort_values(years[-1], ascending=False).head(top_n).to_string())
    for col in changes.columns:
        top = changes[col].abs().sort_values(ascending=False).head(top_n).index
        print(f"\n  Biggest share changes, {col} ({label}):")
        print(changes.loc[top, col].round(2).to_string())