import keras
import tensorflow as tf

system_sparse_categorical_cross_entropy = keras.losses.SparseCategoricalCrossentropy()

def custom_sparse_categorical_cross_entropy(y_true, y_pred):
    y_pred_probs = tf.gather_nd(y_pred, tf.expand_dims(y_true, axis=1), batch_dims=1)
    cross_entropy = -1 * keras.ops.log(y_pred_probs)
    loss = keras.ops.mean(cross_entropy)
    return loss

y_true = tf.constant([1,2])
y_pred = tf.Variable([[0.05, 0.90, 0.05], [0.1, 0.8, 0.1]], dtype="float32")
assert(custom_sparse_categorical_cross_entropy(y_true, y_pred) == system_sparse_categorical_cross_entropy(y_true, y_pred))

x = [[1,2], [3,4]]
print(type(x))
x_t = tf.constant(x)


def loss_fn(y_true, y_pred):
    return keras.ops.sum(-1 * (y_true * keras.ops.log(y_pred) + (1.0 -y_true) * keras.ops.log(1.0 - y_pred)))

l = loss_fn(tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), tf.constant([0.0,0.24,0.90,0.11,0.33,0.45]))

print(l)
