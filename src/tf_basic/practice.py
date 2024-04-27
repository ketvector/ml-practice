import tensorflow as tf

rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

x = tf.Variable([1,2,3,4])
print(x)

with tf.GradientTape() as tape:
    y = tf.matmul(x, x)
    print(y)
    dy_dx = tape.gradient(y,x)
    print(dy_dx)

