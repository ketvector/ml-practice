import pandas as pd
from sklearn.model_selection import train_test_split
import keras_nlp
import keras
import os.path
import numpy as np
import tensorflow as tf

VERBOSE = True
USE_SAVED_MODEL = False
SAVED_MODEL_LOCATION = "./model.keras"
BASE_PATH = "./" #"/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"

def get_prediction_from_ordinal(ord):
    sum = 0
    i = 0
    while(ord[i] >= 0.5):
        sum = sum + 1
        i = i + 1
    return sum


def count_occurences(y, num_classes):
    counts = np.zeros(shape=(num_classes,))
    for item in y.numpy():
        val = get_prediction_from_ordinal(item)
        counts[val] = counts[val] + 1
    #print("counts", counts)
    return counts 

class QuadraticWeightedKappaMetric(keras.metrics.Metric):
    def __init__(self, num_classes, name='quadratic_weighted_kappa', **kwargs):
        super().__init__(name=name, **kwargs)
        self.O = self.add_weight(shape=(num_classes,num_classes), initializer='zeros', dtype='float32')
        self.actual_counts = self.add_weight(shape=(num_classes,), initializer='zeros', dtype='float32')
        self.predicted_counts = self.add_weight(shape=(num_classes,), initializer='zeros', dtype='float32')
        self.w = self.add_weight(shape=(num_classes,num_classes), initializer='zeros', dtype='float32')
        w = np.zeros(shape=(num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                w[i][j] = ((i - j)**2)/((num_classes-1)**2)
        self.w.assign(tf.constant(w))
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.actual_counts.assign_add(count_occurences(y_true, self.num_classes))
        self.predicted_counts.assign_add(count_occurences(y_pred, self.num_classes))
        update = np.zeros(shape=(self.num_classes, self.num_classes))
        for y_t, y_p in zip(y_true, y_pred):
            val_y_t = get_prediction_from_ordinal(y_t)
            val_y_p = get_prediction_from_ordinal(y_p)
            update[val_y_t - 1][val_y_p - 1] = update[val_y_t - 1][val_y_p - 1] + 1
        self.O.assign_add(self.w * update)
    
    def result(self):
        E = keras.ops.outer(self.actual_counts, self.predicted_counts)
        normal_E = E / keras.ops.sum(E)
        normal_O = self.O / keras.ops.sum(self.O)
        print("self.actual_counts", self.actual_counts.numpy())
        print("self.predicted_counts", self.predicted_counts.numpy())
        fraction = keras.ops.sum(normal_O) / keras.ops.clip(keras.ops.sum(self.w * normal_E), 1e-10,  1e25)
        return 1 -fraction

    def reset_state(self):
        self.O.assign(keras.ops.zeros(shape=(self.num_classes, self.num_classes)))
        self.actual_counts.assign(keras.ops.zeros(shape=(self.num_classes)))
        self.predicted_counts.assign(keras.ops.zeros(shape=(self.num_classes)))
        w = np.zeros(shape=(self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                w[i][j] = ((i - j)**2)/((self.num_classes-1)**2)
        self.w.assign(tf.constant(w))

    

def print_(*args):
    if(VERBOSE):
        print(args)

def convert_sparse_value_to_ordinal(val):
    base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(val):
        base[i] = 1.0
    return base
    

def get_train_data():
    return get_data("train")

def get_test_data():
    return get_data("test")


def get_data(type_):
    # Path to the CSV file
    csv_file = f'{BASE_PATH}/train.csv' if type_ == "train" else f'{BASE_PATH}/test.csv'

    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(csv_file)

    if(type_ == "train"):
        df["score"] = df["score"] - 1

    # Get information about the dataframe
    df_info = df.info()
    # Print the information
    print_("dataframe info", df_info)


    # Print the shape of the training set
    print_("{type} set shape is {shape}:".format(type=type_,shape=df.shape))

    train_df_subset = df

    # Extract the "full_text" values as X
    X = train_df_subset["full_text"].values

    if(type_ == "train"):
        # Extract the "label" values as y
        y = train_df_subset["score"].apply(convert_sparse_value_to_ordinal).values
        print(type(list(y)))
        return X, tf.constant(list(y))
    else:
        return X

def get_model():
    if(USE_SAVED_MODEL and os.path.isfile(SAVED_MODEL_LOCATION)):
        return keras.models.load_model(SAVED_MODEL_LOCATION)
    
    else:
        return keras_nlp.models.BertClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=6,
        )
    
def loss_fn(y_true, y_pred):
    return keras.ops.mean(-1 * (y_true * keras.ops.log(keras.ops.clip(y_pred, 1e-10, 1.0)) + (1.0 -y_true) * keras.ops.log(keras.ops.clip(1.0 - y_pred, 1e-10, 1.0))))



def main():
    print("is executing eagerly ?", tf.executing_eagerly())
    model = get_model()
    model.compile(optimizer=model.optimizer, loss = loss_fn, metrics=[QuadraticWeightedKappaMetric(6)], run_eagerly=True)
    X_train, y_train = get_train_data()
    print_("model summary: ", model.summary())
    # Fit the classifier on the training data
    print(X_train[0])
    print(y_train[0])
    model.fit(X_train, y_train, batch_size=50, epochs=3)
    X_test = get_test_data()
    predictions = model.predict(X_test)
    print(predictions, type(predictions))
    predictions = np.apply_along_axis(get_prediction_from_ordinal, axis=1, arr=predictions)
    print(predictions)

    sub_df = pd.read_csv(f"{BASE_PATH}/test.csv")
    sub_df["score"] = predictions
    sub_df = sub_df.drop("full_text", axis=1)
    sub_df.head()
    sub_df.to_csv("submission.csv")

main()


