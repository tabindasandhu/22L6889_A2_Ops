import sys
import platform
import tensorflow as tf
import numpy as np
import pandas as pd
import PIL
import sklearn
import mlflow

print("Python      :", sys.version.split()[0])
print("OS          :", platform.platform())
print("TensorFlow  :", tf.__version__)
print("NumPy       :", np.__version__)
print("Pandas      :", pd.__version__)
print("Pillow      :", PIL.__version__)
print("scikit-learn:", sklearn.__version__)
print("MLflow      :", mlflow.__version__)
print("GPUs visible:", tf.config.list_physical_devices("GPU"))