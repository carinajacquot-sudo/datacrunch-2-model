"""
Fabrique de modèles — chaque modèle est un appelable(Xtr, ytr, Xte, m).

Toutes les variantes implémentent la même signature, donc elles sont
interchangeables dans l'évaluateur et dans le soumissionnaire.
"""
import numpy as np


def _lgb(**kw):
    import lightgbm as lgb
    p = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
             min_child_samples=300, colsample_bytree=0.3, subsample=0.8,
             subsample_freq=1, reg_lambda=5.0, n_jobs=-1, verbose=-1)
    p.update(kw)
    return lgb.LGBMRegressor(**p)


def _lgbc(**kw):
    import lightgbm as lgb
    p = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
             min_child_samples=300, colsample_bytree=0.3, subsample=0.8,
             subsample_freq=1, reg_lambda=5.0, n_jobs=-1, verbose=-1)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def rank01(p):
    n = len(p)
    return (np.argsort(np.argsort(p)) + 0.5) / n if n else p


# --------------------------------------------------------------------------- #
def m_brut(Xtr, ytr, Xte, m):
    """Référence : LightGBM régression sur la cible brute."""
    g = _lgb().fit(Xtr, ytr)
    return g.predict(Xte)


def m_deux_etages(Xtr, ytr, Xte, m):
    """P(non nul) × E[cible | non nul]. Exploite les 87 % de zéros."""
    nz = ytr != 0
    if nz.sum() < 500:
        return m_brut(Xtr, ytr, Xte, m)
    clf = _lgbc().fit(Xtr, nz.astype(int))
    p_nz = clf.predict_proba(Xte)[:, 1]
    reg = _lgb(min_child_samples=100).fit(Xtr[nz], ytr[nz])
    return p_nz * reg.predict(Xte)


def m_pondere(Xtr, ytr, Xte, m):
    """Mêmes données, mais les lignes informatives comptent 8× plus."""
    w = np.where(ytr != 0, 8.0, 1.0)
    g = _lgb().fit(Xtr, ytr, sample_weight=w)
    return g.predict(Xte)


def m_quantile(Xtr, ytr, Xte, m):
    """Prédit la médiane conditionnelle : robuste aux queues extrêmes."""
    g = _lgb(objective="quantile", alpha=0.5).fit(Xtr, ytr)
    return g.predict(Xte)


def m_hurdle_clf(Xtr, ytr, Xte, m):
    """Classifier directement le SIGNE des mouvements significatifs."""
    big = (np.abs(ytr) > 0.5).astype(int)
    if big.sum() < 500 or big.sum() == len(big):
        return m_brut(Xtr, ytr, Xte, m)
    clf = _lgbc(min_child_samples=200).fit(Xtr, big)
    return clf.predict_proba(Xte)[:, 1]


def m_ensemble(Xtr, ytr, Xte, m):
    """Moyenne des rangs de 3 modèles complémentaires (variance réduite)."""
    ps = [rank01(m_brut(Xtr, ytr, Xte, m)),
          rank01(m_deux_etages(Xtr, ytr, Xte, m)),
          rank01(m_pondere(Xtr, ytr, Xte, m))]
    return np.mean(ps, axis=0)


def m_lgb_dart(Xtr, ytr, Xte, m):
    """DART : régularisation par dropout d'arbres, souvent meilleure hors échantillon."""
    g = _lgb(boosting_type="dart", n_estimators=300).fit(Xtr, ytr)
    return g.predict(Xte)


def m_extra_trees(Xtr, ytr, Xte, m):
    """ExtraTrees : très forte décorrélation avec LightGBM → utile en ensemble."""
    from sklearn.ensemble import ExtraTreesRegressor
    g = ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_leaf=100,
                            max_features=0.3, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    return g.predict(Xte)


REGISTRY = {
    "brut":        m_brut,
    "deux_etages": m_deux_etages,
    "pondere":     m_pondere,
    "quantile":    m_quantile,
    "hurdle_clf":  m_hurdle_clf,
    "ensemble":    m_ensemble,
    "lgb_dart":    m_lgb_dart,
    "extra_trees": m_extra_trees,
}
