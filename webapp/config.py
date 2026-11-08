"""All settings come from environment variables (with safe local defaults)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()                                            # reads the project's .env file
BASE_DIR = Path(__file__).resolve().parent.parent        # project root folder


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "visionguard_champion.keras"))
    MODEL_META_PATH = os.getenv("MODEL_META_PATH", str(BASE_DIR / "models" / "visionguard_champion.json"))
    DECISION_THRESHOLD = os.getenv("DECISION_THRESHOLD")          # optional override of the json value
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'visionguard.db').as_posix()}")
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    DEBUG = _as_bool(os.getenv("FLASK_DEBUG", "0"))               # OFF unless explicitly turned on
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
