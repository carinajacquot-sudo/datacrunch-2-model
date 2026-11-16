# La machine de guerre — DataCrunch #2

```
machine/
├── config.py         une seule source de vérité (contraintes, seuils)
├── evaluate_model.py l'évaluateur unique : walk-forward + verdict
├── models.py         8 modèles interchangeables, même signature
├── selector.py       choisit 4 modèles DÉCORRÉLÉS (limite du compte)
└── submit.py         soumission avec garde-fous
```

## Règle unique de la machine

**Aucun modèle ne sort sans avoir passé `verdict()`.** Deux conditions :

- corrélation moyenne ≥ **0,015**
- |t-stat| ≥ **2,0**

En dessous, la machine refuse de gaspiller une soumission. Un modèle qui
« a l'air bon » sans preuve statistique ne sort pas.

## Ce que `submit.py` refuse de faire

| Garde-fou | Raison |
|---|---|
| soumettre sans vérifier les 8 contrôles du scoring officiel | un run non conforme est rejeté |
| soumettre après la clôture du dimanche 12:00 UTC | un run qui dépasse est ignoré |
| dépasser 5 soumissions/jour | limite officielle |
| soumettre avec un t-stat < 2 | on ne parie pas sur du bruit |

```bash
python3 machine/submit.py --check      # diagnostic, n'envoie rien
python3 machine/submit.py --go         # soumet (exige CRUNCH_TOKEN)
```

## Les contraintes réelles, vérifiées dans la doc

| Contrainte | Valeur | Source |
|---|---|---|
| modèles par compte | **4** | `projectCreationLimit` de l'API |
| soumissions par jour | **5** | `resources-limit.md` |
| quota de calcul | **15 h/semaine** | page compétition |
| clôture | **dimanche 12:00 UTC** | page compétition |
| premier score | **semaine 6** | 1 j + 4 semaines de résolution |
| embargo | **4 moons** | `runner.py` |

## Les 8 modèles

| Nom | Idée | Pourquoi ça peut marcher |
|---|---|---|
| `brut` | LightGBM sur la cible brute | référence |
| `deux_etages` | P(non nul) × E[cible \| non nul] | exploite les **87,18 %** de zéros |
| `pondere` | lignes nulles à poids 1, non nulles à 8 | variante simple du précédent |
| `quantile` | régression de la médiane | robuste aux queues |
| `hurdle_clf` | classifier le signe des gros mouvements | cible binaire, plus facile |
| `ensemble` | moyenne des rangs de 3 modèles | réduit la variance |
| `lgb_dart` | dropout d'arbres | régularisation différente |
| `extra_trees` | forêt aléatoire extrême | très décorrélé de LightGBM |

## La stratégie à 4 modèles

On a le droit à **4 modèles**. Le forum CrunchDAO le dit : *« Multiple model =
more reward and better meta-model »*. Mais soumettre 4 fois le même modèle ne
sert à rien : `selector.py` impose une corrélation mutuelle **< 0,95**.
