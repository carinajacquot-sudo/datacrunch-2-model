"""
Sélection des modèles à soumettre.

Contrainte : max MODELS_PER_ACCOUNT (4) modèles, et ils doivent être
DÉCORRÉLÉS — soumettre 4 fois le même modèle ne rapporte rien.
"""
import itertools
import numpy as np

from .config import MODELS_PER_ACCOUNT


def pick_diverse(preds: dict, k: int = MODELS_PER_ACCOUNT, max_corr: float = 0.95):
    """
    preds : {nom: prédictions (1D)} sur les mêmes lignes.
    Garde les k modèles les plus performants dont la corrélation mutuelle
    reste sous max_corr, en favorisant d'abord le meilleur.
    """
    names = list(preds)
    if not names:
        return []
    C = np.corrcoef(np.vstack([preds[n] for n in names]))
    chosen = [names[0]]
    for i, n in enumerate(names[1:], start=1):
        if len(chosen) >= k:
            break
        if all(abs(C[i, names.index(c)]) < max_corr for c in chosen):
            chosen.append(n)
    return chosen


def best_subset(preds: dict, scores: dict, k: int = MODELS_PER_ACCOUNT):
    """Meilleur sous-ensemble de taille k : maximise la somme des scores sous contrainte de décorrélation."""
    k = min(k, len(preds))
    best, best_val = (), -np.inf
    for combo in itertools.combinations(preds, k):
        P = np.vstack([preds[c] for c in combo])
        if np.abs(np.corrcoef(P)).max() > 0.98:
            continue
        val = sum(scores[c] for c in combo)
        if val > best_val:
            best, best_val = combo, val
    return list(best) if best else list(preds)[:k]
