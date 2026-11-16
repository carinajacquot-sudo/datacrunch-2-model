"""
Configuration centrale — une seule source de vérité.
Tout le reste de la machine lit ce fichier.
"""

COMPETITION = "datacrunch-2"
KAGGLE_DATASET = "jacquotvelomanana/datacrunch2-data"

# --- contraintes de la plateforme (vérifiées dans la doc officielle) ---
SUBMISSIONS_PER_DAY = 5          # resources-limit.md
MODELS_PER_ACCOUNT = 4           # projectCreationLimit de l'API
QUOTA_HOURS_WEEK = 15            # quota de calcul
CLOSING_WEEKDAY = 6              # dimanche
CLOSING_HOUR_UTC = 12            # 12:00 UTC
FIRST_SCORE_WEEKS = 6            # 1 jour + 4 semaines de résolution

# --- contrat du scoring officiel (scoring.py::check) ---
REQUIRED_COLUMNS = {"id", "moon", "prediction"}
PRED_MIN, PRED_MAX = -10.0, 10.0
ID_DTYPE_MUST_BE_INT = True
EMBARGO = 4                      # runner.py : EMBARGO = 4

# --- structure du jeu de données (mesurée, pas supposée) ---
N_FEATURES = 1150
N_MOONS_TOTAL = 130              # variante small
N_MOONS_TRAIN = 772              # variante complète
TARGET_ZERO_FRACTION = 0.8718    # 87,18 % des cibles sont exactement 0
FEATURES_ARE_UNIFORM_RANKS = True

# --- seuils de décision ---
MIN_CORR_TO_SUBMIT = 0.015       # en dessous, on ne gaspille pas une soumission
MIN_TSTAT_TO_SUBMIT = 2.0        # exige une preuve statistique, pas une impression
