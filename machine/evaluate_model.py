#!/usr/bin/env python3
"""
Évaluateur unique : la seule façon de juger un modèle.

Usage (dans le notebook Kaggle ou en local) :
    from machine.evaluate_model import walk_forward, verdict
    res = walk_forward(predict_fn, X, y, moon, n_test=20)
    verdict(res)

Règles :
  · walk-forward strict : on n'entraîne que sur moon <= m - EMBARGO
  · métrique = Pearson cross-sectionnelle par moon (le score exact)
  · le hasard est TOUJOURS calculé pour servir de témoin
  · verdict refuse un modèle dont le t-stat < MIN_TSTAT_TO_SUBMIT
"""
import numpy as np

from .config import EMBARGO, MIN_CORR_TO_SUBMIT, MIN_TSTAT_TO_SUBMIT


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def walk_forward(predict_fn, X, y, moon, n_test=20, embargo=EMBARGO, min_rows=5000):
    """predict_fn(Xtr, ytr, Xte, m) -> prédictions. Renvoie le tableau des corrélations."""
    moons = np.unique(moon)
    test_moons = moons[-n_test:]
    out = []
    for m in test_moons:
        te = np.flatnonzero(moon == m)
        tr = np.flatnonzero(moon <= m - embargo)
        if len(tr) < min_rows or len(te) < 30:
            continue
        p = predict_fn(X[tr], y[tr], X[te], m)
        out.append(pearson(p, y[te]))
    return np.array(out)


def random_baseline(y, moon, n_test=20, embargo=EMBARGO):
    def f(Xtr, ytr, Xte, m):
        return np.random.default_rng(int(m)).normal(size=len(Xte))
    return walk_forward(f, np.zeros((len(y), 1), np.float32), y, moon, n_test, embargo)


def verdict(cors, name="modèle", baseline=None):
    import json
    cors = np.asarray(cors, float)
    n = len(cors)
    mean = cors.mean()
    sd = cors.std(ddof=1) if n > 1 else float("nan")
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else 0.0
    out = {
        "modele": name, "n_moons": n, "moyenne": round(float(mean), 5),
        "mediane": round(float(np.median(cors)), 5),
        "part_positifs": round(float((cors > 0).mean()), 3),
        "t_stat": round(float(t), 2),
        "ic95": [round(float(mean - 1.96 * sd / np.sqrt(n)), 5),
                 round(float(mean + 1.96 * sd / np.sqrt(n)), 5)],
    }
    if baseline is not None:
        out["edge_sur_hasard"] = round(float(mean - np.mean(baseline)), 5)
    ok_corr = mean >= MIN_CORR_TO_SUBMIT
    ok_t = abs(t) >= MIN_TSTAT_TO_SUBMIT
    out["soumettre"] = bool(ok_corr and ok_t)
    out["raison"] = (
        "OK : corrélation et t-stat au-dessus des seuils"
        if out["soumettre"] else
        f"REFUS : corr {mean:+.4f} (seuil {MIN_CORR_TO_SUBMIT}) / t {t:.2f} (seuil {MIN_TSTAT_TO_SUBMIT})"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out
