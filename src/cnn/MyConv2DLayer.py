import tensorflow as tf
import numpy as np
import utils
import fileloader

class MyConv2DLayer(tf.Module):
    def __init__(self, name="None", num_kernels = 1, kernel_size = 1):
        super().__init__(name=name)
        self.kernels = tf.Variable(tf.random.normal(shape=(num_kernels, kernel_size , kernel_size), mean=0.0, stddev=1.0, dtype=tf.float64))
    def __call__(self, x):
        conv_rows = x.shape[1] - self.kernels.shape[1] + 1
        conv_cols = x.shape[2] - self.kernels.shape[2] + 1
        num_kernels = self.kernels.shape[0]
        conv_result = tf.Variable(tf.zeros(shape=(x.shape[0], num_kernels, conv_rows, conv_cols ), dtype=tf.float64))
        for item, index in zip(x, range(x.shape[0])):
            #print("index " , index)
            conv_result_item = tf.Variable(tf.zeros(shape=(num_kernels,conv_rows, conv_cols), dtype=tf.float64))
            for k in range(num_kernels):
                for r in range(conv_rows):
                    for c in range(conv_cols):
                        temp = utils.local_conv(item,  self.kernels[k], r, c)
                        conv_result_temp = tf.scatter_nd(tf.constant([[k, r, c]]), updates=tf.expand_dims(temp,axis=0) , shape=tf.constant([num_kernels,conv_rows, conv_cols]))
                        conv_result_item = conv_result_item + conv_result_temp
            #print(conv_result_item[0])
            update = tf.scatter_nd(tf.constant([[index]]), updates=tf.expand_dims(conv_result_item, axis=0), shape=tf.constant([x.shape[0],num_kernels,conv_rows, conv_cols]))
            conv_result = conv_result + update
        return conv_result

def test():
    train_x_np, train_y, test_x_np, test_y = fileloader.get_data(3)
    train_x = utils.wrap_in_tf(train_x_np)
    conv_layer = MyConv2DLayer(num_kernels=4, kernel_size=5)
    with tf.GradientTape() as tape:
        conv_layer_output = conv_layer(train_x)
        loss = tf.reduce_mean(conv_layer_output)
        print(loss)

    gradient = tape.gradient(loss, conv_layer.kernels)
    print(gradient)

#test()




