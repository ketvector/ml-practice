import keras
import tensorflow as tf
import numpy as np
from  utils import *

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
            update[val_y_t][val_y_p] = update[val_y_t][val_y_p] + 1
        self.O.assign_add(update)
    
    def result(self):
        E = keras.ops.outer(self.actual_counts, self.predicted_counts)
        # debug_("self.actual_counts", self.actual_counts.numpy())
        # debug_("self.predicted_counts", self.predicted_counts.numpy())
        # debug_("E" , E)
        # debug_("O", self.O.numpy())
        normal_E = E / keras.ops.sum(E)
        normal_O = self.O / keras.ops.sum(self.O)
        fraction = keras.ops.sum(self.w * normal_O) / keras.ops.sum(self.w * normal_E)
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