"""
Task 3.6 - Load the registered model by its 'champion' alias and classify one image.
Usage:  python load_champion.py path/to/image.jpg
"""
import sys

import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()                                   # DagsHub tracking URI + token from .env
MODEL_NAME = "22L6889_A2_Ops_VisionGuard"
image_path = sys.argv[1]

client = MlflowClient()
version = client.get_model_version_by_alias(MODEL_NAME, "champion")
threshold = float(version.tags.get("decision_threshold", 0.5))
img_size = int(version.tags.get("img_size", 224))
print(f"Champion = version {version.version} | threshold {threshold} | image size {img_size}")

model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}@champion")    # load BY ALIAS

# EXACTLY the same preprocessing as training: decode RGB -> resize with pad -> float
img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
img = tf.image.resize_with_pad(img, img_size, img_size)
x = tf.expand_dims(tf.cast(img, tf.float32), axis=0)                      # batch of 1

p_ai = float(model.predict(x, verbose=0)[0][0])
label = "AI-generated" if p_ai >= threshold else "Real"
confidence = p_ai if label == "AI-generated" else 1 - p_ai
print(f"{image_path}\n -> {label}  (P(AI) = {p_ai:.3f}, confidence = {confidence:.1%})")
