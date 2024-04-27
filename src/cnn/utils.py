import tensorflow as tf
import numpy as np

def wrap_in_tf(n_np, dtype=tf.float64):
     return tf.constant(n_np, dtype=dtype)

def local_conv(input, kernel, r, c):
        l = input[r : r + kernel.shape[0], c : c + kernel.shape[1]]
        r = kernel
        return tf.reduce_sum(tf.multiply(l,r))

def loss(y, class_, pred_probs):
    loss_ = tf.Variable(0.0, dtype = tf.float64)
    for actual_y, predicted_y, predicted_value_for_actual_y in zip(y, class_, pred_probs):
          if (actual_y != predicted_y):
               loss_ = loss_.assign_add(tf.multiply(tf.cast(tf.constant(-1.0), tf.float64), tf.math.log(predicted_value_for_actual_y)))
          else:
               loss_ = loss_.assign_add(tf.Variable(0.0, dtype=tf.float64))
    return loss_
                  
                 
        