"""Database table: one row per prediction (what, when, which model, how confident)."""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True,
                           default=lambda: datetime.now(timezone.utc))      # WHEN
    filename = db.Column(db.String(255))                                     # sanitised name only
    label = db.Column(db.String(20), nullable=False)                         # WHAT was decided
    confidence = db.Column(db.Float, nullable=False)                         # HOW confidently
    p_ai = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), nullable=False)                   # BY WHICH model
    model_version = db.Column(db.String(20), nullable=False)
    latency_ms = db.Column(db.Float)
