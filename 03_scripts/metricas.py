"""Métricas compartidas (AUC macro, umbrales, P/R/F1, bootstrap). Sin dependencia de torch."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import config as C

# =============================================================================================
# Métricas
# =============================================================================================
def safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def macro_auc(Y, P):
    return float(np.nanmean([safe_auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


def best_thresholds(Y, P):
    """Umbral por hallazgo que maximiza F1 (se elige en VALIDACIÓN débil, nunca en dorados)."""
    ths = []
    for j in range(Y.shape[1]):
        grid = np.unique(np.quantile(P[:, j], np.linspace(0.02, 0.98, 49)))
        if Y[:, j].sum() == 0 or len(grid) == 0:
            ths.append(0.5)
            continue
        f1s = [f1_score(Y[:, j], P[:, j] >= t, zero_division=0) for t in grid]
        ths.append(float(grid[int(np.argmax(f1s))]))
    return np.array(ths)


def per_label_report(Y, P, ths):
    rows = []
    for j, lab in enumerate(C.LABELS):
        b = (P[:, j] >= ths[j]).astype(int)
        rows.append({"hallazgo": lab, "n_pos": int(Y[:, j].sum()), "auc": safe_auc(Y[:, j], P[:, j]),
                     "umbral": float(ths[j]),
                     "precision": precision_score(Y[:, j], b, zero_division=0),
                     "recall": recall_score(Y[:, j], b, zero_division=0),
                     "f1": f1_score(Y[:, j], b, zero_division=0)})
    return pd.DataFrame(rows)


def bootstrap_ci(Y, P, n=1000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    vals = [macro_auc(Y[idx], P[idx]) for idx in (rng.integers(0, len(Y), len(Y)) for _ in range(n))]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
