import numpy as np
from  constants import *


def get_prediction_from_ordinal(ord):
    sum = 0
    i = 0
    while(i < len(ord) and ord[i] >= 0.5 ):
        sum = sum + 1
        i = i + 1
    return sum


def count_occurences(y, num_classes):
    counts = np.zeros(shape=(num_classes,))
    for item in y.numpy():
        val = get_prediction_from_ordinal(item)
        counts[val] = counts[val] + 1
    return counts

def debug_(*args):
    if(VERBOSE):
        print(args)

def convert_sparse_value_to_ordinal(val):
    base = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(val):
        base[i] = 1
    return base