import pandas as pd
from  constants import *
from  utils import *
import tensorflow as tf

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
    debug_("dataframe info", df_info)


    # Print the shape of the training set
    debug_("{type} set shape is {shape}:".format(type=type_,shape=df.shape))

    train_df_subset = df

    # Extract the "full_text" values as X
    X = train_df_subset["full_text"].values

    if(type_ == "train"):
        # Extract the "label" values as y
        y = train_df_subset["score"].apply(convert_sparse_value_to_ordinal).values
        debug_(type(list(y)))
        return X, tf.constant(list(y))
    else:
        return X

def get_train_data():
    return get_data("train")

def get_test_data():
    return get_data("test")