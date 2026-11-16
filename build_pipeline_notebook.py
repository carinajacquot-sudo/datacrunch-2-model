"""Construit le notebook de test du pipeline en injectant le VRAI main.py."""
import json
import pathlib

MAIN = pathlib.Path('/home/user/dc2/main.py').read_text()
CLEAN = MAIN.replace('"""', "'''")   # evite de casser la cellule magique

code = "%%writefile main.py\n" + CLEAN

test = r'''
import os, sys, json, time, gc, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import pearsonr

T0 = time.time()
def el(): return f"{time.time()-T0:.0f}s"

hits = glob.glob("/kaggle/input/**/X.reduced.parquet", recursive=True)
DATA = os.path.dirname(hits[0])
print("DATA =", DATA)

pf = pq.ParquetFile(hits[0])
y = pd.read_parquet(glob.glob(os.path.join(DATA, "y*.parquet"))[0])
y = y.sort_values(["moon", "id"]).reset_index(drop=True)
moon_all = y["moon"].to_numpy()
uniq = np.unique(moon_all)
print(f"{pf.metadata.num_columns-2} features · {len(y):,} lignes · {len(uniq)} moons · {el()}")

sys.path.insert(0, "/kaggle/working")
import main as M
print("recette :", M.PARAMS["n_estimators"], "arbres ·", M.TOPK, "features · refit tous les", M.REFIT_EVERY, "moons")

mdl = "/kaggle/working/resources"
os.makedirs(mdl, exist_ok=True)
scores, times = [], []
test_moons = list(uniq[-12:])

for m in test_moons:
    tr_rows = np.flatnonzero(moon_all <= m - 4)
    te_rows = np.flatnonzero(moon_all == m)

    X = pf.read().to_pandas()
    xtr, ytr = X.iloc[tr_rows], y.iloc[tr_rows][["moon", "id", "target"]]
    t0 = time.time()
    M.train(xtr, ytr, mdl, loop_moon=int(m), embargo=4)
    times.append(time.time() - t0)
    xte = X.iloc[te_rows]
    del X, xtr; gc.collect()

    out = M.infer(xte, mdl, loop_moon=int(m), embargo=4)

    assert list(out.columns) == ["moon", "id", "prediction"]
    assert len(out) == len(te_rows) == out["id"].nunique()
    assert out["id"].dtype == np.int64 and out["moon"].dtype == np.int64
    assert np.isfinite(out["prediction"]).all()
    assert out["prediction"].between(-10, 10).all()
    assert out["prediction"].nunique() > 1, f"prediction constante au moon {m}"

    yte = y.iloc[te_rows]["target"].to_numpy(np.float64)
    r = float(pearsonr(out["prediction"].to_numpy(), yte)[0])
    scores.append(r)
    print(f"  moon {m}: {len(out)} lignes · r={r:+.4f} · {time.time()-t0:.0f}s · {el()}", flush=True)
    del xte, out; gc.collect()

s = np.array(scores); n = len(s)
res = {
    "n_moons": n,
    "moyenne": round(float(s.mean()), 5),
    "mediane": round(float(np.median(s)), 5),
    "part_positifs": round(float((s > 0).mean()), 3),
    "t_stat": round(float(s.mean() / (s.std(ddof=1) / np.sqrt(n))), 2),
    "ic95": [round(float(s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(n)), 5),
             round(float(s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(n)), 5)],
    "temps_train_moyen_s": round(float(np.mean(times)), 1),
    "projection_781_moons_h": round(float(np.mean(times)) * 781 / 2 / 3600, 2),
    "contrat_respecte": True,
    "scores": [round(float(x), 5) for x in scores],
}
print(json.dumps(res, indent=1))
json.dump(res, open("/kaggle/working/pipeline_results.json", "w"), indent=1)

X = pf.read().to_pandas()
t0 = time.time()
M.train(X, y[["moon", "id", "target"]], mdl, loop_moon=int(uniq[-1]) + 1, embargo=4)
print(f"modele final : {time.time()-t0:.0f}s · {os.path.getsize(os.path.join(mdl,'model.txt'))/1e6:.1f} Mo")
json.dump({"ok": True}, open("/kaggle/working/DONE.json", "w"))
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Test du pipeline de soumission\n", "\n",
            "Le **vrai `main.py`** est injecte ci-dessous, puis execute dans une replique\n",
            "exacte de la boucle du runner : `train()` puis `infer()` moon par moon.\n", "\n",
            "On verifie trois choses : le contrat de sortie, le score reel, et le cout\n",
            "en calcul projete sur 781 moons."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": test},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}

json.dump(nb, open('/home/user/kaggle_pipe/pipeline_test.ipynb', 'w'), indent=1)
print("notebook ecrit ·", len(MAIN), "caracteres injectes")
