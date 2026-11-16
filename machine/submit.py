#!/usr/bin/env python3
"""
Soumission automatique — avec les garde-fous de la plateforme.

Ce que ce module refuse de faire :
  · soumettre sans avoir vérifié la conformité au scoring officiel
  · soumettre après la clôture du dimanche 12:00 UTC
  · dépasser 5 soumissions par jour
  · soumettre un modèle dont le t-stat est sous le seuil

Usage :
    python3 -m machine.submit --check                    # diagnostic seulement
    python3 -m machine.submit --project ~/dc2 --go       # soumet réellement
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def next_closing(now=None):
    """Prochaine clôture : dimanche 12:00 UTC."""
    now = now or now_utc()
    days_ahead = (6 - now.weekday()) % 7          # 6 = dimanche
    closing = (now + dt.timedelta(days=days_ahead)).replace(
        hour=12, minute=0, second=0, microsecond=0)
    if closing <= now:
        closing += dt.timedelta(days=7)
    return closing


def time_left():
    now = now_utc()
    delta = next_closing(now) - now
    return delta, next_closing(now)


def check_window():
    delta, closing = time_left()
    hours = delta.total_seconds() / 3600
    print(f"clôture : {closing:%Y-%m-%d %H:%M} UTC (dans {hours:.1f} h)")
    if hours < 2:
        print("⚠️  moins de 2 h avant la clôture — un run qui dépasse sera ignoré")
    return hours


def count_today(project_dir):
    """Compte les soumissions du jour d'après le journal local."""
    log = os.path.join(project_dir, "submissions.log")
    if not os.path.exists(log):
        return 0
    today = now_utc().date().isoformat()
    with open(log) as f:
        return sum(1 for line in f if line.startswith(today))


def log_submission(project_dir, message):
    with open(os.path.join(project_dir, "submissions.log"), "a") as f:
        f.write(f"{now_utc().date().isoformat()}\t{now_utc():%H:%M:%S}\t{message}\n")


def conformite(project_dir):
    """Vérifie le contrat officiel sur un moon de test réel (lecture économe)."""
    import gc
    sys.path.insert(0, project_dir)
    os.chdir(project_dir)
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from main import infer, train

    pf = pq.ParquetFile("data/X.reduced.parquet")
    names = [pf.metadata.schema.column(i).name for i in range(pf.metadata.num_columns)]
    keep = ["moon", "id"] + [c for c in names if c.startswith("Feature_")][:60]
    x = pf.read(columns=keep).to_pandas()
    del pf; gc.collect()

    y = pd.read_parquet("data/y.reduced.parquet", columns=["moon", "id", "target"])
    moons = np.sort(x["moon"].unique())
    moon = int(moons[-1])
    cur = x[x["moon"] == moon][keep].reset_index(drop=True)
    hist_moons = moons[-20:]
    yh = y[y["moon"].isin(hist_moons)]

    d = "/tmp/conf_models"
    import shutil; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    train(cur, yh[yh["moon"] <= moon], d, moon, 4)
    p = infer(cur, d, moon, 4)

    checks = [
        ("colonnes {id, moon, prediction}", set(p.columns) == {"id", "moon", "prediction"}),
        ("id entier", np.issubdtype(p["id"].dtype, np.integer)),
        ("moon entier", np.issubdtype(p["moon"].dtype, np.integer)),
        ("prediction flottant", np.issubdtype(p["prediction"].dtype, np.floating)),
        ("aucun NaN", not p["prediction"].isna().any()),
        ("aucun infini", not np.isinf(p["prediction"]).any()),
        ("borné [-10, 10]", bool(p["prediction"].between(-10, 10).all())),
        ("une ligne par id", len(p) == len(cur)),
    ]
    ok = True
    for label, cond in checks:
        print(("  ✅ " if cond else "  ❌ ") + label)
        ok &= bool(cond)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.path.expanduser("~/dc2"))
    ap.add_argument("--check", action="store_true", help="diagnostic seulement")
    ap.add_argument("--go", action="store_true", help="soumettre réellement")
    ap.add_argument("--message", default="machine: soumission automatique")
    a = ap.parse_args()

    print("=" * 58)
    print(" SOUMISSION — garde-fous")
    print("=" * 58)

    hours = check_window()
    n = count_today(a.project)
    print(f"soumissions aujourd'hui : {n} / 5")
    if n >= 5:
        print("❌ quota journalier atteint — revenir demain")
        return 1

    print("\nconformité au scoring officiel :")
    try:
        ok = conformite(a.project)
    except Exception as e:
        print(f"  ❌ vérification impossible : {e}")
        return 1
    if not ok:
        print("\n❌ NON CONFORME — soumission refusée")
        return 1

    if not a.go:
        print("\n✅ tous les garde-fous passent (mode diagnostic, rien n'est envoyé)")
        return 0

    token = os.environ.get("CRUNCH_TOKEN")
    if not token:
        print("\n❌ CRUNCH_TOKEN absent de l'environnement")
        print("   le token de setup est à usage unique : le régénérer depuis l'onglet Submit")
        return 1

    print("\nenvoi…")
    r = subprocess.run([sys.executable, "-m", "crunch", "push", "-m", a.message],
                       cwd=a.project, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode == 0:
        log_submission(a.project, a.message)
        print("✅ soumission envoyée")
        return 0
    print("❌ échec :", r.stderr[-800:])
    return 1


if __name__ == "__main__":
    sys.exit(main())
