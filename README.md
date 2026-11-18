# DataCrunch Equity Market Neutral #2 — pipeline de soumission

**Compétition :** DataCrunch #2 (CrunchDAO / Crunch Capital) · 1 000 USDC/semaine
**Clôture :** dimanche 12:00 UTC · quota 15 h de calcul/semaine
**Statut :** `unranked` — aucune soumission effectuée à ce jour

---

## Le modèle livré

`main.py` implémente **une seule recette, choisie parce qu'elle a été mesurée**,
pas parce qu'elle était disponible.

### `D_peu_arbres` — validé bout en bout

Deux mesures, deux méthodes, même conclusion :

| Mesure | Score | t | Moons positifs |
|---|---|---|---|
| Campagne `la-verite` — criblage sur toutes les lignes d'entraînement, 24 moons | **+0,0188** | 3,97 | 88 % |
| **Pipeline de soumission** — criblage échantillonné, 12 moons | **+0,0267** | **7,42** | **100 %** |

La seconde est celle qui compte : c'est le code réel, exécuté dans une
réplique exacte de la boucle du runner, `train()` puis `infer()` moon après
moon. Contrat de sortie vérifié, 2,17 h de calcul projetées pour un budget
de 15 h, 26,4 Go de pic mémoire.

**Pourquoi le pipeline marque plus** : les moons de test du pipeline ne sont
pas centrés (le runner ne fournit que les features, pas la cible), donc le
modèle prédit aussi un écart de niveau entre moons — ce que le scoring
orthogonalise ensuite. Ce n'est pas un gain réel, c'est une différence de
protocole. Le chiffre prudent reste **+0,0188**.

C'est le meilleur des **53 configurations** testées.

| Étape | Choix | Pourquoi |
|---|---|---|
| Cible | centrée par moon puis σ=1 par moon | la cible est déjà neutralisée, on retire le niveau de moon |
| Features | **250** criblées par \|corr\| moyenne par moon | optimum mesuré : 1 150 → +0,0063, 250 → +0,0188, 40 → +0,0063 |
| Criblage | **dans `train()` seulement** | sur toutes les données il fabriquait une fuite : +0,0146 annoncé, +0,0014 réel |
| Modèle | LightGBM **100 arbres**, lr 0,05, leaves 15, min_child 300, colsample 0,3, subsample 0,8, λ=5 | avec un signal faible, plus d'arbres mémorise le bruit : 300 → +0,0124, 600 → +0,0113 |
| Sortie | **rangs uniformes (0,1) par moon** | la métrique est un rang ; Pearson sur valeur = +0,0031, sur rang = +0,0159 |

### Ce qui a été écarté, et pourquoi

| Famille | Meilleur score | t | Verdict |
|---|---|---|---|
| linéaires (ridge, lasso, elasticnet, PCA) | +0,0089 | 1,77 | ne franchit pas t=2, lasso à +0,0000 |
| LambdaRank / LTR | +0,0127 | 2,16 | pas assez de paires par moon |
| deux étages (P(non nul) × E[cible]) | +0,0109 | 2,34 | la classe « non nul » est trop rare et bruitée |
| pondération par l'ampleur (Jane Street) | +0,0068 | 1,32 | ne se transpose pas ici |
| cascade de résidus | +0,0039 | 0,87 | le résidu est du bruit |

Le registre complet des 40 modèles est dans
[`datacrunch-2-knowledge`](https://github.com/carinajacquot-sudo/datacrunch-2-knowledge)
→ `registry/modeles.json` et `07-resultats-mesures.md`.

---

## Budget de calcul — le piège qui annule une soumission

Le runner estime le temps par moon **avant** de lancer. Si l'estimation dépasse
le temps restant, il coupe au milieu :

> *stopping early at moon=791 because the estimated required time (~192h23m45s)
> is greater than the remaining time before timeout (~14h57m32s)*

Un modèle trop lourd ne donne pas un mauvais score : il donne **aucun score**.

`train()` mesure sa propre durée et applique cette politique :

| Réentraînement | Coût total (781 moons) | Verdict |
|---|---|---|
| à chaque moon | ~2,6 h | acceptable mais sans marge |
| **tous les 2 moons** | **~1,3 h** | **retenu** |
| tous les 4 moons | ~0,7 h | si l'entraînement dépasse 45 s |

Le modèle est réutilisé tant qu'il a moins de 2 moons, via `resources/state.json`.
Le criblage se fait par blocs de 200 colonnes en float32 pour tenir en mémoire.

---

## Conformité au contrat

`infer()` produit exactement `{moon, id, prediction}` et vérifie :

- `moon` et `id` en `int64`, `prediction` en `float64`
- une ligne par `id`, aucun `NaN`, aucun infini
- `prediction ∈ [-10, 10]`
- **aucune prédiction constante par moon** (interdit → score 0)
- rangs calculés **à l'intérieur** de chaque moon, jamais entre moons

---

## Structure

```
main.py                  le modèle livré (train + infer, contrat CrunchDAO)
kernel-metadata.json     déclaration du kernel Kaggle qui teste le pipeline
pipeline_test.ipynb      réplique exacte de la boucle du runner sur 12 moons
machine/                 outils d'analyse locaux (évaluation, sélection, soumission)
```

## Ce qui reste à faire

1. Le kernel `datacrunch-2-pipeline-test` valide le contrat et le coût sur Kaggle.
2. Lire les résultats de `idees-publiques` (statistiques transversales de ligne,
   quintiles par moon) — la dernière famille d'idées non testée.
3. Soumettre, puis suivre le premier score (semaine 6 : 1 jour + 4 semaines).

## Historique des mesures

| Date | Événement |
|---|---|
| 15/11 | fuite de sélection détectée : +0,0146 annoncé → +0,0014 réel |
| 16/11 | `la-verite` : 9 configurations sans fuite, gagnant +0,0188 (t 3,97) |
| 16/11 | `recherche-2` : familles linéaires toutes battues ⇒ axe fermé |
| 16/11 | `astuces-js` : poids temporels +0,0123, autres astuces sous-performantes |
| 16/11 | registre consolidé : 40 modèles |

---

*Le code ne contient que du code : les données, les jetons et les modèles
sérialisés ne sont jamais versionnés (voir `.gitignore`).*
