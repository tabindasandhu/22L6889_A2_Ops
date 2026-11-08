"""Loads the champion model ONCE and turns uploaded image bytes into a verdict."""
import json
import time

import tensorflow as tf


class Predictor:
    def __init__(self, model_path, meta_path, threshold_override=None):
        start = time.time()
        self.model = tf.keras.models.load_model(model_path, compile=False)   # heavy: done once
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.img_size = int(self.meta.get("img_size", self.model.input_shape[1]))
        self.threshold = float(threshold_override) if threshold_override else float(self.meta["decision_threshold"])
        self.model_name = self.meta.get("model_name", "VisionGuard")
        self.model_version = str(self.meta.get("model_version", "unknown"))
        self.load_seconds = time.time() - start

    def preprocess(self, image_bytes):
        """EXACTLY the training pipeline: decode RGB -> resize with pad -> float. Scaling is inside the model."""
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        img = tf.image.resize_with_pad(img, self.img_size, self.img_size)
        return tf.expand_dims(tf.cast(img, tf.float32), axis=0)               # batch of 1

    def predict(self, image_bytes):
        start = time.perf_counter()
        p_ai = float(self.model(self.preprocess(image_bytes), training=False).numpy()[0][0])
        label = "AI-generated" if p_ai >= self.threshold else "Real"
        confidence = p_ai if label == "AI-generated" else 1.0 - p_ai
        return {"label": label, "confidence": confidence, "p_ai": p_ai,
                "threshold": self.threshold, "model_name": self.model_name,
                "model_version": self.model_version,
                "latency_ms": (time.perf_counter() - start) * 1000}


if __name__ == "__main__":                     # console test: python inference.py img1 img2
    import sys
    from config import Config

    predictor = Predictor(Config.MODEL_PATH, Config.MODEL_META_PATH, Config.DECISION_THRESHOLD)
    print(f"Model v{predictor.model_version} loaded in {predictor.load_seconds:.1f}s "
          f"(threshold {predictor.threshold}, image size {predictor.img_size})")
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            r = predictor.predict(f.read())
        print(f"{path}\n  -> {r['label']} | confidence {r['confidence']:.1%} | "
              f"P(AI) {r['p_ai']:.3f} | {r['latency_ms']:.0f} ms")
