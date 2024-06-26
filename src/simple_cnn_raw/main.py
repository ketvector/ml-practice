import fileloader
from MyConv2DLayer import MyConv2DLayer
from MyClassificationLayer import MyClassificationLayer
import utils
import tensorflow as tf
import numpy as np


"""
  Simple two layer convolutional neural network. One Conv layer, One classification layer.

  Written using raw tensorflow (no keras, module etc), for learning tensorflow.

  Does classification on the MNIST dataset.

"""


train_x_np, train_y_np, test_x_np, test_y = fileloader.get_data(2)
train_x = utils.wrap_in_tf(train_x_np)
test_x = utils.wrap_in_tf(test_x_np)
train_y = utils.wrap_in_tf(train_y_np, dtype=tf.int64)

learning_rate = tf.constant(0.1, dtype=tf.float64)

def train(x, y, conv_layer, class_layer):
    with tf.GradientTape() as tape:
        conv_layer_output = conv_layer(x)
        flat = tf.linalg.normalize(tf.reshape(conv_layer_output,(conv_layer_output.shape[0],-1)), axis=1)[0]
        class_, probs = class_layer(flat)
        print("probs", probs)
        print("class", class_)
        print("y", y)
        range_ = tf.convert_to_tensor(range(tf.shape(y)[0]))
        pred_probs = tf.gather_nd(probs, tf.stack([range_, tf.cast(y, tf.int32)], axis=1))
        print(pred_probs)
        current_loss = utils.loss(y,  class_, pred_probs)
        print("current_loss", current_loss)
    
    dkernel, dweights = tape.gradient(current_loss, [conv_layer.kernels, class_layer.weights])
    print(dkernel, dweights)



conv_layer = MyConv2DLayer(num_kernels=4, kernel_size=5)
class_layer = MyClassificationLayer(num_classes=10, inputs_to_each_class= 4 * 24 * 24)

conv_layer_checkpoint = tf.train.Checkpoint(conv_layer)
class_layer_checkpoint = tf.train.Checkpoint(class_layer)

train(train_x, train_y, conv_layer, class_layer )