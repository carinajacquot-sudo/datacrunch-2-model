"""Analyse de structure : les 1150 features sont-elles organisées en blocs ?
Si oui, on peut construire des ratios/écarts DANS un bloc — ce que personne ne fait."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, gc

pf = pq.ParquetFile("data/X.reduced.parquet")
names = [pf.metadata.schema.column(i).name for i in range(pf.metadata.num_columns)]
feats = [c for c in names if c.startswith("Feature_")]
idx = {f: int(f.split("_")[1]) for f in feats}

y = pd.read_parquet("data/y.reduced.parquet").sort_values(["moon","id"]).reset_index(drop=True)
moon = y["moon"].to_numpy(np.int32)
uniq, starts, counts = np.unique(moon, return_index=True, return_counts=True)
gm = np.repeat(np.arange(len(uniq)), counts)

# 1) corrélation entre features ADJACENTES (indices voisins) par moon
print("=== corrélation moyenne entre features voisines (i, i+1) ===")
pairs = []
B = 120
for i in range(0, len(feats), B):
    blk = feats[i:i+B]
    x = pf.read(columns=blk).to_pandas()[blk].to_numpy(np.float32)
    x -= (np.add.reduceat(x, starts, axis=0)/counts[:,None])[gm]
    sd = np.sqrt(np.add.reduceat(x**2, starts, axis=0)); sd[sd==0]=np.nan
    z = x/sd[gm]                                   # standardisation par moon
    # corrélation intra-bloc (produit scalaire de z, moyenne sur moons)
    n = counts[:,None]
    c = np.add.reduceat(z[:,:-1]*z[:,1:], starts, axis=0)/n
    pairs.append(np.nanmean(c, axis=0))
    del x, sd, z, c; gc.collect()
adj = np.concatenate(pairs)
print(f"  {len(adj)} paires voisines | moyenne {np.mean(adj):+.4f} | médiane {np.median(adj):+.4f}")
print(f"  |corr| > 0.3 : {(np.abs(adj)>0.3).sum()} paires")
print(f"  |corr| > 0.5 : {(np.abs(adj)>0.5).sum()} paires")
top = np.argsort(-np.abs(adj))[:12]
print("  paires les plus liées :")
for t in top:
    print(f"    Feature_{t+1} ↔ Feature_{t+2} : {adj[t]:+.3f}")

# 2) statistiques de distribution sur un échantillon
print("\n=== distribution (10 features au hasard) ===")
rng = np.random.default_rng(0)
samp = [feats[i] for i in rng.choice(len(feats), 10, replace=False)]
s = pf.read(columns=samp).to_pandas()
print(s.describe().T[["mean","std","min","50%","max"]].to_string(float_format=lambda v: f"{v:12.4f}"))
print(f"\nNaN dans l'échantillon : {int(s.isna().sum().sum())}")
print(f"valeurs infinies : {int(np.isinf(s.to_numpy()).sum())}")
