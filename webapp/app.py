"""VisionGuard web app: upload -> validate -> predict -> log to DB -> show result."""
import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image
from sqlalchemy import text
from werkzeug.utils import secure_filename

from config import Config
from db import Prediction, db
from inference import Predictor

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_BYTES = int(Config.MAX_UPLOAD_MB * 1024 * 1024)

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = Config.DATABASE_URL
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES * 4        # hard safety cap; the friendly check is below

db.init_app(app)
with app.app_context():
    db.create_all()                                      # create the table if it doesn't exist

predictor = Predictor(Config.MODEL_PATH, Config.MODEL_META_PATH, Config.DECISION_THRESHOLD)  # ONCE


class UploadError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def validate_upload(file):
    """Returns (safe_filename, image_bytes) or raises UploadError with a friendly message."""
    if file is None or file.filename == "":
        raise UploadError("Please choose an image file to upload.")

    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:                                                  # 1. wrong type
        raise UploadError(f"'.{ext or '?'}' files are not supported. Please upload a JPG or PNG image.", 415)
    safe_name = secure_filename(file.filename) or f"upload.{ext}"                      # sanitise name

    data = file.read()                                                                 # memory only, never disk
    if not data:
        raise UploadError("The uploaded file is empty.")
    if len(data) > MAX_BYTES:                                                          # 2. too big
        raise UploadError(f"The file is {len(data) / 1048576:.1f} MB. The limit is {Config.MAX_UPLOAD_MB:g} MB.", 413)

    try:                                                                               # 3. can't be decoded
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
    except Exception:
        raise UploadError("This file has an image extension but could not be decoded. "
                          "It may be corrupted or not a real image.", 422)
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"The file's real content is {fmt}, not JPG or PNG.", 415)
    return safe_name, data


@app.get("/")
def index():
    return render_template("index.html", max_mb=Config.MAX_UPLOAD_MB)


@app.post("/predict")
def predict():
    try:
        safe_name, data = validate_upload(request.files.get("image"))
    except UploadError as e:
        return render_template("error.html", title="Upload rejected", message=e.message), e.status

    result = predictor.predict(data)
    record = Prediction(filename=safe_name, label=result["label"], confidence=result["confidence"],
                        p_ai=result["p_ai"], threshold=result["threshold"],
                        model_name=result["model_name"], model_version=result["model_version"],
                        latency_ms=result["latency_ms"])
    db.session.add(record)
    db.session.commit()                                   # permanent record of this decision
    return render_template("result.html", r=result, filename=safe_name, record=record)


@app.get("/history")
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
    return render_template("history.html", rows=rows)


@app.get("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {type(e).__name__}"
    healthy = db_status == "ok" and predictor.model is not None
    return jsonify(status="ok" if healthy else "degraded", model_loaded=predictor.model is not None,
                   model_name=predictor.model_name, model_version=predictor.model_version,
                   threshold=predictor.threshold, database=db_status), (200 if healthy else 503)


@app.errorhandler(413)
def too_large(_e):
    return render_template("error.html", title="File too large",
                           message=f"The limit is {Config.MAX_UPLOAD_MB:g} MB."), 413


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", title="Page not found",
                           message="That page doesn't exist."), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("error.html", title="Something went wrong",
                           message="An unexpected error happened. Please try again."), 500


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
