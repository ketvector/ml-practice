import fileloader
from MyConv2DLayer import MyConv2DLayer
from MyClassificationLayer import MyClassificationLayer
import utils
import tensorflow as tf
import numpy as np

test_x_np, test_y_np = fileloader.get_data(10000)

test_x = utils.wrap_in_tf(test_x_np)
test_y = utils.wrap_in_tf(test_y_np, dtype=tf.uint32)

conv_layer = MyConv2DLayer(num_kernels=4, kernel_size=5)
class_layer = MyClassificationLayer(num_classes=10, inputs_to_each_class= 4 * 24 * 24)

conv_layer_checkpoint = tf.train.Checkpoint(conv_layer)
class_layer_checkpoint = tf.train.Checkpoint(class_layer)

conv_layer_checkpoint.restore('./checkpoints/conv')
class_layer_checkpoint.restore('./checkpoints/class')

def test(conv_layer, class_layer, x, y ):
    conv_layer_output = conv_layer(x)
    class_layer_output, probs = class_layer(tf.reshape(conv_layer_output,(1,tf.size(conv_layer_output))))
    pred_prob = probs[0][class_layer_output[0][0]]
    print(pred_prob, y)