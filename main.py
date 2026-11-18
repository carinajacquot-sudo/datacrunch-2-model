"""
DataCrunch Equity Market Neutral #2 — soumission
================================================

Hôte : Crunch Capital · 1 000 USDC/semaine · clôture dimanche 12:00 UTC

Ce fichier câble le modèle qui a **mesuré le meilleur score sans fuite** :
`D_peu_arbres` — +0,0188 (t = 3,97, 88 % de moons positifs) sur 24 moons
de test jamais vus, embargo 4.

Ce n'est pas un choix par défaut : 48 configurations ont été mesurées.
La fiche complète est dans `registry/modeles.json`.

Recette (identique à celle mesurée, à la ligne près)
----------------------------------------------------
1. Cible centrée cross-sectionnellement par moon, puis normalisée à
   écart-type 1 par moon.
2. Criblage des 250 features les plus corrélées à la cible, **calculé
   uniquement dans `train()`, donc uniquement sur les moons passés**.
3. LightGBM 100 arbres, lr 0,05, num_leaves 15, min_child_samples 300,
   colsample 0,3, subsample 0,8, reg_lambda 5.
4. Prédiction ramenée en rangs uniformes (0,1) **par moon**.

Discipline mémoire — ce qui a failli faire échouer la soumission
----------------------------------------------------------------
L'historique complet fait 1,64 M lignes × 1 152 colonnes :
**6,9 Go en float32, 13,8 Go en float64**. Le runner dispose de ~16 Go.
La version naïve ci-dessous mourait donc avant même d'entraîner :

    df = x_train.merge(y_train, on=["moon", "id"])   # ← copie de 6,9 Go
    X  = x_train[cols].to_numpy()                    # ← seconde copie

Deux règles en découlent, appliquées partout ici :

  · **jamais de merge.** Les `id` sont globalement uniques (1 637 276 ids
    distincts pour 1 637 276 lignes, vérifié) : une simple table de
    correspondance `id → target` remplace la jointure, pour 12 Mo.
  · **jamais de copie de la matrice entière.** Le criblage se fait par
    blocs de 200 colonnes (1,2 Go transitoire), puis on ne matérialise
    que les 250 colonnes retenues (1,4 Go).

Deux fois « Kernel died » sans message avant que je trouve ça. Un noyau
qui meurt silencieusement sur Kaggle, c'est une OOM, pas un bug de logique.
"""

import json
import os
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constantes — toutes issues de mesures, aucune choisie au hasard
# --------------------------------------------------------------------------
TARGET = "target"
META = ("moon", "id")

TOPK = 250              # optimum mesuré (1 150 → +0,0063 · 250 → +0,0188 · 40 → +0,0063)
BLOCK = 200             # colonnes criblées à la fois : 1,2 Go transitoire
MIN_ROWS_TRAIN = 5_000  # en dessous, on s'abstient plutôt que de prédire au hasard

# Recette D_peu_arbres — exactement celle du kernel `la-verite`
PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=300,
    colsample_bytree=0.3,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=5.0,
    n_jobs=-1,
    verbose=-1,
)

# --- Politique de réentraînement -----------------------------------------
# Le runner dispose de 15 h pour TOUT l'historique et coupe net quand son
# estimation dépasse le temps restant. Le coût qui compte est donc
# durée × 781 moons, pas la durée d'un entraînement.
#   à chaque moon ...... 781 × ~12 s ≈ 2,6 h  (sans marge)
#   tous les 2 moons ... 391 × ~12 s ≈ 1,3 h  ← retenu
#   tous les 4 moons ... 196 × ~12 s ≈ 0,7 h
REFIT_EVERY = 2          # entraînement : tous les 2 moons
SCREEN_EVERY = 10        # criblage des features : tous les 10 moons
REFIT_EVERY_SLOW = 4
SLOW_TRAIN_SECONDS = 120.0

# Le criblage est de loin l'étape la plus chère (1 150 colonnes × 1,6 M lignes).
# On l'estime sur un échantillon déterministe de SCREEN_ROWS_PER_MOON lignes par
# moon : la corrélation moyenne par moon est stable, 600 lignes sur ~2 100
# suffisent à classer les features. Le modèle, lui, s'entraîne sur TOUTES les
# lignes : l'approximation ne touche que le classement, pas l'apprentissage.
SCREEN_ROWS_PER_MOON = 600


# --------------------------------------------------------------------------
# Outils
# --------------------------------------------------------------------------
def _rank01(p):
    """Rangs uniformes dans (0, 1). Monotone ⇒ préserve la corrélation."""
    n = len(p)
    if n == 0:
        return p
    return (np.argsort(np.argsort(p)) + 0.5) / n


def _read_state(model_directory_path):
    try:
        return json.load(open(os.path.join(model_directory_path, "state.json")))
    except Exception:
        return {}


def _target_lookup(y_train):
    """id → target. Les id sont globalement uniques : pas besoin de jointure."""
    return pd.Series(
        y_train[TARGET].to_numpy(np.float32),
        index=y_train["id"].to_numpy(np.int64),
    )


def _screen(X, tgt, starts, counts):
    """
    |corr| moyenne par moon, feature par feature, calculée sur les lignes
    fournies uniquement. X est un bloc (n_lignes × n_colonnes ≤ BLOCK).
    """
    blk = np.asarray(X, dtype=np.float32)
    idx = np.repeat(np.arange(len(starts)), counts)      # moon de chaque ligne

    mu = np.add.reduceat(blk, starts, axis=0) / counts[:, None].astype(np.float32)
    blk -= mu[idx]                                       # centrage par moon
    norm = np.sqrt(np.add.reduceat(blk**2, starts, axis=0))
    norm[norm == 0] = np.nan
    c = np.add.reduceat(blk * tgt[:, None], starts, axis=0) / norm
    return np.nanmean(np.abs(c), axis=0).astype(np.float64)


def params_budget(n_rows):
    """Allège le modèle quand l'historique est long, pour tenir le budget de 15 h."""
    p = dict(PARAMS)
    if n_rows > 1_000_000:
        p.update(n_estimators=80, colsample_bytree=0.25)
    return p


def _feature_columns(df):
    return [c for c in df.columns if c not in META and c != TARGET]


# --------------------------------------------------------------------------
# Contrat CrunchDAO
# --------------------------------------------------------------------------
def train(x_train, y_train, model_directory_path, loop_moon, embargo, **kwargs):
    """
    Entraîne sur les moons dont la cible est DÉJÀ résolue : moon ≤ loop_moon − embargo.
    Le criblage est fait ici, et seulement ici ⇒ aucune fuite, et aucune copie
    de la matrice entière.
    """
    import lightgbm as lgb

    t0 = time.time()
    os.makedirs(model_directory_path, exist_ok=True)

    # --- politique de réentraînement -------------------------------------
    state = _read_state(model_directory_path)
    last_moon = state.get("last_train_moon")
    every = (REFIT_EVERY_SLOW if state.get("train_seconds", 0) > SLOW_TRAIN_SECONDS
             else REFIT_EVERY)
    if last_moon is not None and (loop_moon - int(last_moon)) < every:
        print(f"[train] modèle du moon {last_moon} réutilisé (moins de {every} moons)")
        return

    # le criblage est bien plus cher que l'entraînement : on le garde
    # SCREEN_EVERY moons, et on réutilise le classement entre-temps.
    last_screen = state.get("last_screen_moon")
    rescreen = (last_screen is None
                or (loop_moon - int(last_screen)) >= SCREEN_EVERY
                or not os.path.exists(os.path.join(model_directory_path, "feature_ranking.csv")))

    # --- masque d'entraînement, sans jointure ---------------------------
    moons = x_train["moon"].to_numpy()
    ids = x_train["id"].to_numpy()

    lookup = _target_lookup(y_train)
    tgt_all = lookup.reindex(ids).to_numpy(np.float32)     # 6 Mo, pas 6,9 Go
    del lookup

    # une seule liste d'index : les lignes passées ET résolues
    rows = np.flatnonzero(~np.isnan(tgt_all) & (moons <= (loop_moon - embargo)))
    n_rows = len(rows)
    if n_rows < MIN_ROWS_TRAIN:
        print(f"[train] historique insuffisant ({n_rows} lignes) — modèle non écrit")
        return

    # cible centrée par moon puis σ=1 par moon (identique au kernel la-verite)
    tgt = tgt_all[rows].astype(np.float64)
    del tgt_all

    m_tr = moons[rows]
    uniq, starts, counts = np.unique(m_tr, return_index=True, return_counts=True)
    tgt -= np.repeat(np.add.reduceat(tgt, starts) / counts, counts)
    var = np.add.reduceat(tgt**2, starts) / counts
    tgt /= np.repeat(np.sqrt(np.maximum(var, 1e-12)), counts)
    del m_tr

    # --- criblage par blocs de colonnes, sur le passé seul ---------------
    cols = _feature_columns(x_train)
    rank_path = os.path.join(model_directory_path, "feature_ranking.csv")

    if rescreen:
        # on crible sur un échantillon déterministe de lignes par moon : la
        # corrélation moyenne par moon est stable, 600 lignes suffisent à classer.
        keep_rows, keep_tgt, sub_counts = [], [], []
        for k in range(len(uniq)):
            a = starts[k]
            b = a + counts[k]
            if counts[k] > SCREEN_ROWS_PER_MOON:
                step = np.linspace(0, counts[k] - 1, SCREEN_ROWS_PER_MOON).astype(np.int64)
                sel_k = rows[a + step]
                tgt_k = tgt[a + step]
            else:
                sel_k = rows[a:b]
                tgt_k = tgt[a:b]
            keep_rows.append(sel_k)
            keep_tgt.append(tgt_k)
            sub_counts.append(len(sel_k))

        s_rows = np.concatenate(keep_rows)
        s_tgt = np.concatenate(keep_tgt).astype(np.float64)
        s_starts = np.concatenate([[0], np.cumsum(sub_counts)[:-1]])
        s_counts = np.asarray(sub_counts, np.int64)

        abss = np.empty(len(cols), np.float64)
        for i in range(0, len(cols), BLOCK):
            sub = cols[i : i + BLOCK]
            blk = x_train[sub].to_numpy(dtype=np.float32)[s_rows]
            abss[i : i + BLOCK] = _screen(blk, s_tgt, s_starts, s_counts)
            del blk
        del keep_rows, keep_tgt, s_rows, s_tgt

        order = np.argsort(-abss)[:TOPK]
        keep = [cols[j] for j in order]
        abscorr = abss[order]
        del abss

        pd.DataFrame({"feature": keep, "abs_corr": abscorr}).to_csv(rank_path, index=False)
        screened_now = True
        print(f"[train] criblage refait sur {len(s_counts)} moons × "
              f"{SCREEN_ROWS_PER_MOON} lignes · {time.time() - t0:.0f}s", flush=True)
    else:
        keep = pd.read_csv(rank_path)["feature"].tolist()[:TOPK]
        screened_now = False
        print(f"[train] criblage du moon {last_screen} réutilisé ({len(keep)} features)")

    print(f"[train] {n_rows:,} lignes · {len(counts)} moons (≤ {loop_moon - embargo}) "
          f"· {len(keep)}/{len(cols)} features · {time.time() - t0:.0f}s")

    # --- entraînement ----------------------------------------------------
    # On donne les VRAIS noms de features a LightGBM. Avec un tableau numpy
    # nu, le booster sauvegarde Column_0..Column_249, et infer() ne retrouve
    # alors aucune colonne dans les donnees : prediction vide, score nul.
    X = pd.DataFrame(x_train[keep].to_numpy(dtype=np.float32)[rows], columns=keep)
    model = lgb.LGBMRegressor(**params_budget(n_rows))
    model.fit(X, tgt)
    del X

    model.booster_.save_model(os.path.join(model_directory_path, "model.txt"))

    dur = time.time() - t0
    json.dump({"last_train_moon": int(loop_moon), "train_seconds": round(dur, 2),
               "last_screen_moon": int(loop_moon) if screened_now else last_screen,
               "n_rows": n_rows, "n_features": len(keep)},
              open(os.path.join(model_directory_path, "state.json"), "w"))
    print(f"[train] modèle écrit · {dur:.0f}s · réentraînement tous les {every} moons")
    return


def infer(x_test, model_directory_path, loop_moon, embargo, **kwargs):
    """
    Prédit chaque moon et renvoie des rangs uniformes dans (0,1).
    Format : {moon, id, prediction} — conforme au scoring officiel.
    """
    path = os.path.join(model_directory_path, "model.txt")
    moons = x_test["moon"].to_numpy(np.int64)
    ids = x_test["id"].to_numpy(np.int64)
    pred = np.zeros(len(x_test), dtype=np.float64)

    if os.path.exists(path):
        import lightgbm as lgb

        booster = lgb.Booster(model_file=path)
        need = booster.feature_name()

        # filet de securite : un modele sauvegarde avant ce correctif porte des
        # noms generiques (Column_0...). On retrouve alors l'ordre des features
        # dans le classement ecrit a cote du modele.
        if all(c.startswith("Column_") for c in need):
            rank_path = os.path.join(model_directory_path, "feature_ranking.csv")
            if os.path.exists(rank_path):
                need = pd.read_csv(rank_path)["feature"].tolist()[:len(need)]
                print(f"[infer] noms generiques detectes — ordre relu dans feature_ranking.csv")

        cols = [c for c in need if c in x_test.columns]

        if len(cols) == len(need):
            X = x_test[cols].to_numpy(dtype=np.float32)
            raw = np.asarray(booster.predict(X), dtype=np.float64)
            del X
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            # rang À L'INTÉRIEUR de chaque moon : le score compare des titres
            # entre eux au sein d'un même moon, jamais entre moons.
            for m in np.unique(moons):
                s = moons == m
                pred[s] = _rank01(raw[s])
        else:
            print(f"[infer] {len(need) - len(cols)} features absentes — repli neutre")
    else:
        print("[infer] aucun modèle — repli neutre")

    pred = np.clip(pred, -10.0, 10.0)

    out = pd.DataFrame({
        "moon": moons.astype(np.int64),
        "id": ids.astype(np.int64),
        "prediction": pred.astype(np.float64),
    })

    for m in np.unique(moons):
        if out.loc[out.moon == m, "prediction"].nunique() < 2:
            print(f"[infer] ATTENTION moon {m} : prédiction constante")

    return out
