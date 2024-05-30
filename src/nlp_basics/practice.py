import os
import keras
import keras_nlp



# os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"
# Use mixed precision to speed up all training in this guide.

keras.mixed_precision.set_global_policy("mixed_float16")