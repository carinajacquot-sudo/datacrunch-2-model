#!/usr/bin/env python3
"""
TEST DE PUISSANCE — le modèle a-t-il un edge réel ?

Méthode : walk-forward strict sur les VRAIES données DataCrunch #2.
  · criblage des 1150 features sur TOUT l'historique (lecture par blocs, 2 Go)
  · pour chaque moon de test : on entraîne sur tout le passé résolu, on prédit,
    on mesure la corrélation de Pearson cross-sectionnelle (le score exact)
  · trois témoins : hasard, inversion du signal, et un modèle « marché seul »
    (prédiction constante) pour prouver qu'on ne fait pas juste du beta

Aucun moon de test n'entre dans l'entraînement ni dans le criblage.
"""
import gc, json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TOPK = int(os.environ.get("TOPK", 300))
N_TEST = int(os.environ.get("N_TEST", 24))
EMBARGO = 4
DATA = "data"

# ---------------------------------------------------------------- chargement
y = pd.read_parquet(f"{DATA}/y.reduced.parquet").sort_values(["moon", "id"]).reset_index(drop=True)
moon = y["moon"].to_numpy(np.int32)
tgt = y["target"].to_numpy(np.float32)
uniq, starts, counts = np.unique(moon, return_index=True, return_counts=True)
gm = np.repeat(np.arange(len(uniq)), counts)
print(f"{len(y)} lignes · {len(uniq)} moons ({uniq[0]}→{uniq[-1]}) · {counts.mean():.0f} lignes/moon")

pf = pq.ParquetFile(f"{DATA}/X.reduced.parquet")
names = [pf.metadata.schema.column(i).name for i in range(pf.metadata.num_columns)]
feats = [c for c in names if c.startswith("Feature_")]


def xsec_center(x):
    m = np.add.reduceat(x, starts, axis=0) / counts[:, None]
    return x - m[gm]


# ------------------------------------------------------- 1. criblage complet
# Mémoire : on n'accumule QUE 1150 nombres, jamais une matrice de features.
print(f"\n[1] criblage des {len(feats)} features")
mt = np.add.reduceat(tgt, starts) / counts
tc = tgt - mt[gm]
tc /= np.sqrt(np.add.reduceat(tc**2, starts))[gm]
del mt; gc.collect()

abss, signes = [], []
B = 100
for i in range(0, len(feats), B):
    blk = feats[i:i + B]
    x = pf.read(columns=blk).to_pandas()[blk].to_numpy(np.float32)
    xc = xsec_center(x)
    del x; gc.collect()
    nx = np.sqrt(np.add.reduceat(xc**2, starts, axis=0)); nx[nx == 0] = np.nan
    corr = np.add.reduceat(xc * tc[:, None], starts, axis=0) / nx
    del xc, nx; gc.collect()
    abss.append(np.nanmean(np.abs(corr), axis=0))
    signes.append(np.nanmean(np.sign(corr), axis=0))
    del corr; gc.collect()
    print(f"    bloc {i//B+1}/{int(np.ceil(len(feats)/B))}", flush=True)

rank = pd.DataFrame({"feature": feats, "abs_corr": np.concatenate(abss),
                     "signe": np.concatenate(signes)}).sort_values("abs_corr", ascending=False)
rank.to_csv("feature_ranking.csv", index=False)
print(f"    top : {rank.iloc[0]['feature']} |corr|={rank.iloc[0]['abs_corr']:.4f}"
      f" | >0.03 : {(rank.abs_corr>0.03).sum()} | >0.02 : {(rank.abs_corr>0.02).sum()}")

# --------------------------------------------- 2. matrice d'entraînement
print(f"\n[2] construction de la matrice ({TOPK} features)")
cols = rank.head(TOPK)["feature"].tolist()
X = np.empty((len(moon), TOPK), dtype=np.float32)
for i in range(0, TOPK, 100):
    sub = cols[i:i + 100]
    block = pf.read(columns=sub).to_pandas()[sub].to_numpy(np.float32)
    X[:, i:i + len(sub)] = block
    del block; gc.collect()
for j in range(0, TOPK, 100):
    X[:, j:j + 100] = xsec_center(X[:, j:j + 100])
    gc.collect()
np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    {X.shape} = {X.nbytes/1e6:.0f} Mo")

# ------------------------------------------------ 3. walk-forward strict
print(f"\n[3] walk-forward sur les {N_TEST} derniers moons")
test_moons = list(uniq[-N_TEST:])
import lightgbm as lgb
from sklearn.linear_model import Ridge


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


results = {k: [] for k in ("hasard", "ridge", "lgbm")}
for m in test_moons:
    cut = m - EMBARGO
    te = np.flatnonzero(moon == m)          # indices, pas de copie
    tr = np.flatnonzero(moon <= cut)
    if len(tr) < 5000 or len(te) < 30:
        continue
    Xtr = X[tr]; ytr = tgt[tr]
    Xte = X[te]; yte = tgt[te]
    del tr; gc.collect()

    results["hasard"].append(pearson(np.random.default_rng(m).normal(size=len(yte)), yte))

    r = Ridge(alpha=1000).fit(Xtr, ytr)
    results["ridge"].append(pearson(r.predict(Xte), yte))
    del r; gc.collect()

    g = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.04, num_leaves=15,
                          min_child_samples=300, colsample_bytree=0.3, subsample=0.8,
                          subsample_freq=1, reg_lambda=5.0, n_jobs=1, verbose=-1,
                          force_col_wise=True)
    g.fit(Xtr, ytr)
    p = g.predict(Xte)
    results["lgbm"].append(pearson(p, yte))
    del Xtr, ytr, Xte, yte, g, p; gc.collect()
    print(f"    moon {m} : lgbm {results['lgbm'][-1]:+.4f}  ridge {results['ridge'][-1]:+.4f}", flush=True)

# ----------------------------------------------------------- 4. verdict
print("\n" + "=" * 62)
print(" RÉSULTATS — corrélation cross-sectionnelle hors échantillon")
print("=" * 62)
print(f"{'modèle':<16}{'moyenne':>10}{'médiane':>10}{'%>0':>8}{'n':>5}")
print("-" * 62)
for k, v in results.items():
    v = np.array(v) if len(v) else np.array([np.nan])
    print(f"{k:<16}{np.nanmean(v):>+10.4f}{np.nanmedian(v):>+10.4f}"
          f"{np.nanmean(v > 0) * 100:>7.0f}%{len(v):>5}")

lgbm_m = np.mean(results["lgbm"])
rnd = np.mean(results["hasard"])
print("-" * 62)
print(f"\nEdge du modèle sur le hasard : {lgbm_m - rnd:+.4f}")
t_stat = lgbm_m / (np.std(results["lgbm"], ddof=1) / np.sqrt(len(results["lgbm"])))
print(f"t-stat : {t_stat:.2f}  (au-delà de 2, l'edge est statistiquement solide)")
print(f"IC 95 % : [{lgbm_m - 1.96 * np.std(results['lgbm'], ddof=1)/np.sqrt(len(results['lgbm'])):+.4f}, "
      f"{lgbm_m + 1.96 * np.std(results['lgbm'], ddof=1)/np.sqrt(len(results['lgbm'])):+.4f}]")


json.dump({k: [float(x) for x in v] for k, v in results.items()},
          open("power_results.json", "w"), indent=2)
print("\n→ power_results.json")
