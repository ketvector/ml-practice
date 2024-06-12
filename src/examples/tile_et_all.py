import keras
import tensorflow as tf


def weight():
    num_classes = 6
    label_vec = keras.ops.arange(num_classes, dtype=keras.backend.floatx())
    row_label_vec = keras.ops.reshape(label_vec, [1, num_classes])
    col_label_vec = keras.ops.reshape(label_vec, [num_classes, 1])
    print("row", row_label_vec)
    print("col", col_label_vec)
    col_mat = keras.ops.tile(col_label_vec, [1, num_classes])
    print("col mat", col_mat)
    row_mat = keras.ops.tile(row_label_vec, [num_classes, 1])
    print("row_mat", row_mat)
    weight_mat = (col_mat - row_mat) ** 2
    print(weight_mat)

def mul():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[1,0], [2,3]])
    c = a * b
    print(c.numpy())


#weight()
mul()