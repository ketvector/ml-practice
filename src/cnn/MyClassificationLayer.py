import tensorflow as tf
import numpy as np

class MyClassificationLayer(tf.Module):
     def __init__(self, name="None", num_classes = 10, inputs_to_each_class = 1):
          super().__init__(name)
          self.weights = tf.Variable(tf.random.normal(shape=(inputs_to_each_class + 1, num_classes), mean=0.0, stddev=1.0, dtype=tf.float64))
     def __call__(self, x):
          self.x = x
          x_with_bias = tf.concat((x, tf.constant(tf.ones(shape=(x.shape[0],1), dtype=tf.float64))), axis=1)
          s = tf.matmul(x_with_bias, self.weights)
          exp_s = tf.exp(s)
          denoms = tf.reduce_sum(exp_s, axis=1, keepdims = True)
          probs = exp_s / denoms
          max_index = tf.argmax(probs, axis=1)
          #print("probs ", probs)
          return (max_index, probs)

