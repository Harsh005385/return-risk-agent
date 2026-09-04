from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Re-export for convenience when imports use src.paths
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "src" / "models"
AUDIT_DB = DATA_PROCESSED / "audit.db"
METRICS_PATH = DATA_PROCESSED / "metrics.json"
ARTIFACT_PATH = MODEL_DIR / "model_bundle.joblib"
ABUSE_CSV = DATA_RAW / "ecommerce_return_abuse_dataset.csv"
RETURNS_CSV = DATA_RAW / "returns_sustainability_dataset.csv"
