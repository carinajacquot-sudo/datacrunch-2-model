# DataCrunch #2 — la machine

**Compétition :** DataCrunch Equity Market Neutral #2 (CrunchDAO / Crunch Capital)
**Prix :** 1 000 USDC/semaine · **clôture dimanche 12:00 UTC** · quota 15 h/semaine

## Découvertes qui guident tout le reste

| # | Fait mesuré sur les vraies données | Conséquence |
|---|---|---|
| 1 | **87,18 %** des cibles valent exactement **0** | un modèle naïf gaspille 87 % de sa capacité → modèle à deux étages |
| 2 | les features sont des **rangs uniformes [0,1]**, toutes identiques en distribution | normaliser ne sert à rien |
| 3 | **aucune** corrélation entre features voisines (max 0,001) | elles sont orthogonales, pas de redondance à exploiter |
| 4 | la relation feature→cible est **non linéaire** (profil en U, corr +0,85) | une régression linéaire — le quickstarter officiel — passe à côté |
| 5 | `projectCreationLimit = 4` | jusqu'à **4 modèles** par compte, décorrélés |
| 6 | le token de setup est **à usage unique** (3 min) | il faut le régénérer à chaque soumission |

## Structure

```
machine/
├── config.py           contraintes et seuils, une seule source de vérité
├── evaluate_model.py   walk-forward + verdict statistique
├── models.py           8 modèles interchangeables
├── selector.py         4 modèles décorrélés (< 0,95)
└── submit.py           soumission avec 4 garde-fous
```

## Règle de la machine

Un modèle ne sort que si **corrélation ≥ 0,015 ET |t-stat| ≥ 2,0**.
Sinon `submit.py` refuse. Pas de pari sur du bruit.

## Résultats obtenus

Walk-forward, 20 moons jamais vus, données réelles, 250 features :

| Modèle | Moyenne | % moons positifs | t-stat |
|---|---|---|---|
| hasard | −0,0074 | 40 % | — |
| Ridge | +0,0153 | 65 % | — |
| **LightGBM** | **+0,0146** | **80 %** | **3,20** |

Edge sur le hasard : **+0,0220** · IC 95 % [+0,0057, +0,0236] → edge réel.

## Utilisation

```bash
python3 machine/submit.py --check      # diagnostic complet, n'envoie rien
python3 machine/submit.py --go         # soumet (CRUNCH_TOKEN requis)
```

## Secrets

Aucune clé dans ce dépôt. `kaggle.json`, les tokens CrunchDAO et le token
GitHub sont exclus par `.gitignore`.
