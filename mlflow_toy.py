"""
Task 3.1 - MLflow tool lab (local)
Trains a tiny Keras model on MNIST and logs params, per-epoch metrics,
a plot and the model to a LOCAL MLflow store.
Usage:  python mlflow_toy.py 0.001
"""
import sys

import matplotlib
matplotlib.use("Agg")                    # save plots to file, no window
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import os
from dotenv import load_dotenv

load_dotenv()      # reads .env into environment variables (if the file exists)
# ---------- settings ----------
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001   # learning rate from command line
EPOCHS = 5
BATCH_SIZE = 64
HIDDEN_UNITS = 64
N_TRAIN = 10_000                         # small subset -> fast on CPU

# Use DagsHub if MLFLOW_TRACKING_URI is set (from .env); otherwise fall back to local
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
print("Tracking to:", mlflow.get_tracking_uri())
mlflow.set_experiment("22L6889_A2_Ops_toy")      # group runs together

tf.keras.utils.set_random_seed(42)       # same results every run

# ---------- data ----------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = x_train[:N_TRAIN] / 255.0, y_train[:N_TRAIN]   # scale pixels 0-255 -> 0-1
x_test = x_test / 255.0

# ---------- model ----------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),                         # 28x28 image -> 784 numbers
    tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),   # 10 digit classes
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """After every epoch, send that epoch's metrics to MLflow."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


# ---------- one MLflow run ----------
with mlflow.start_run(run_name=f"lr_{LR}"):
    # parameters = inputs we chose
    mlflow.log_params({
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_units": HIDDEN_UNITS,
        "train_samples": N_TRAIN,
    })
    # tags = labels for organising
    mlflow.set_tags({"task": "3.1", "dataset": "mnist", "purpose": "toy"})

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[MLflowEpochLogger()],
        verbose=2,
    )

    # final test metrics
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

    # artifact 1: a plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"Loss curves (lr={LR})"); ax.legend()
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=120)
    plt.close(fig)
    mlflow.log_artifact("loss_curve.png", artifact_path="plots")

    # artifact 2: the model itself
    mlflow.tensorflow.log_model(model, name="model")

    print(f"\nDone. lr={LR}  test_accuracy={test_acc:.4f}")