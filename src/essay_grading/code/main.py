import pandas as pd
import keras
import numpy as np
import tensorflow as tf
from  utils import debug_, get_prediction_from_ordinal
from  models import get_model, loss_fn
from  quadratic_weighted_kappa import QuadraticWeightedKappaMetric
from  data import get_train_data, get_test_data
from  constants import *

def main():
    debug_("is executing eagerly ?", tf.executing_eagerly())
    debug_("keras version", keras.__version__)
    preprocessor, model = get_model()
    model.compile(optimizer=keras.optimizers.Adam(5e-6), loss = loss_fn, metrics=[QuadraticWeightedKappaMetric(6)], run_eagerly=True)
    model.summary()
    def preprocess_fn(text, label=None):
        text = preprocessor(text)  # Preprocess text
        return (text, label) if label is not None else text  # Return processed text and label if available
    
    X_train, y_train = get_train_data()
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(preprocess_fn)
    train_ds = train_ds.batch(batch_size=50)
    model.fit(train_ds, epochs=3)
    X_test = get_test_data()
    test_ds = tf.data.Dataset.from_tensor_slices(X_test)
    test_ds = test_ds.map(preprocess_fn)
    test_ds = test_ds.batch(50)
    predictions = model.predict(test_ds)
    debug_(predictions, type(predictions))
    predictions = np.apply_along_axis(get_prediction_from_ordinal, axis=1, arr=predictions)
    debug_(predictions)

    sub_df = pd.read_csv(f"{BASE_PATH}/test.csv")
    sub_df["score"] = predictions + 1
    sub_df = sub_df.drop("full_text", axis=1)
    sub_df.head()
    sub_df.to_csv(f"{SAVE_PATH}/submission.csv", index=False)

main()


